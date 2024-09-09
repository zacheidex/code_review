import os
import time
import csv
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from PIL import Image
import pandas as pd


################################################################################################################################################################
class settings:
    total_patients = 506
    num_epochs = 201
    val_step = 5

################################################################################################################################################################

class PNGDataset(Dataset):
    def __init__(self, input_dir_1, csv_file, clinical_csv, set_type, flip_prob=0.5):
        self.input_dir_1 = input_dir_1
        self.flip_prob = flip_prob
        self.set_type = set_type

        with open(csv_file, "r", newline="") as csvfile:
            reader = csv.reader(csvfile)
            self.patient_ids = [row[0] for row in reader if row[1] == set_type]

        self.filtered_file_dict = {}
        for patient_id in self.patient_ids:
            patient_dir_1 = os.path.join(input_dir_1, patient_id)
            patient_files_1 = [f for f in os.listdir(patient_dir_1) if f.endswith('.png')]
            self.filtered_file_dict[patient_id] = list(set(patient_files_1))

        self.clinical_data = pd.read_csv(clinical_csv)

    def __len__(self):
        return sum(len(file_list) for file_list in self.filtered_file_dict.values())

    def random_flip(self, img1, ground_truth):
        for axis in range(2):
            if np.random.random() < self.flip_prob:
                img1 = np.flip(img1, axis=axis).copy()
        return img1, ground_truth

    def get_patient_id_and_filename(self, idx):
        for patient_id, file_list in self.filtered_file_dict.items():
            if idx < len(file_list):
                return patient_id, file_list[idx]
            idx -= len(file_list)

    def __getitem__(self, idx):
        patient_id, file_name = self.get_patient_id_and_filename(idx)
        input_file_1 = os.path.join(self.input_dir_1, patient_id, file_name)
        input_img_1 = np.array(Image.open(input_file_1)) / 255  # Normalize to [0, 1]

        if self.set_type == "train":
            input_img_1, _ = self.random_flip(input_img_1, None)

        input_img = torch.Tensor(np.array([input_img_1]))  # Convert to Tensor

        patient_data = self.clinical_data[self.clinical_data['ID'] == patient_id]

        if not patient_data.empty:
            clinical_vector = patient_data['WHO CNS Grade'].values[0]  # Extract the WHO CNS Grade
            clinical_vector = 1 if clinical_vector > 2 else 0
        else:
            print(f"Warning: {patient_id} not found in clinical data.")

        clinical_tensor = torch.FloatTensor([clinical_vector])
        return input_img, clinical_tensor, file_name

################################################################################################################################################################

def plot_losses(train_losses, val_losses, val_epochs, epoch, save_path):
    plt.figure()
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Training Loss')
    plt.plot(val_epochs, val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Over Time')
    plt.legend()
    plt.savefig(save_path)
    plt.close()

################################################################################################################################################################
def load_date_set(dir, csv_file1, csv_file2, name="train"):
    dataset = PNGDataset(dir, csv_file1, csv_file2, name)
    return dataset
################################################################################################################################################################

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    output_folder = 'resnet50'
    if not os.path.exists('./results/' + output_folder):
        os.makedirs('./results/' + output_folder)

    loss_plot_file = f'./results/{output_folder}/progress.png'
    metrics_file = f'./results/{output_folder}/metrics.csv'
    input_dir_1 = './Datasets/ADC_glioma_downsampled_pngs'
    csv_file = 'patient_ids.csv'
    clinical_csv_file = 'UCSF-data.csv'

    num_epochs = settings.num_epochs
    val_step = settings.num_epochs

    train_dataset = load_date_set(input_dir_1, csv_file, clinical_csv_file, "train")
    val_dataset = load_date_set(input_dir_1, csv_file, clinical_csv_file, "val")

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

    model = models.resnet50(pretrained=True)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    model.fc = nn.Linear(in_features=model.fc.in_features, out_features=1)
    model = nn.Sequential(model, nn.Sigmoid())
    model.to(device)


    total_patients = settings.total_patients
    weight_for_class_1 = total_patients / 456  # Weight for WHO CNS Grade > 2
    weight_for_class_0 = total_patients / 50  # Weight for WHO CNS Grade <= 2

    class_weights = torch.tensor([weight_for_class_0, weight_for_class_1], dtype=torch.float32).to(device)

    # Use the weights in the loss function
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([weight_for_class_1]).to(device))
    optimizer_G = torch.optim.AdamW(model.parameters(), lr=0.0001, betas=(0.5, 0.999), eps=1e-6)

    train_losses = []
    val_losses = []
    val_epochs = []

    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        model.train()
        running_loss = 0.0
        num_batches = 0

        for i, data in enumerate(train_loader, 0):
            inputs, labels, _ = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer_G.zero_grad()
            outputs = model(inputs)
            total_loss = criterion(outputs, labels)
            total_loss.backward()
            optimizer_G.step()

            running_loss += total_loss.item()
            num_batches += 1

        avg_train_loss = running_loss / num_batches
        train_losses.append(avg_train_loss)

        epoch_end_time = time.time()
        epoch_time = (epoch_end_time - epoch_start_time) / 60

        print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Time: {epoch_time:.2f} min')

        if (epoch + 1) % val_step == 0:
            model.eval()
            val_running_loss = 0.0
            val_num_batches = 0
            with torch.no_grad():
                for i, val_data in enumerate(val_loader, 0):
                    val_inputs, val_labels, _ = val_data
                    val_inputs = val_inputs.to(device)
                    val_labels = val_labels.to(device)

                    outputs = model(val_inputs)
                    val_loss = criterion(outputs, val_labels)
                    val_running_loss += val_loss.item()
                    val_num_batches += 1

            avg_val_loss = val_running_loss / val_num_batches
            val_losses.append(avg_val_loss)
            val_epochs.append(epoch + 1)

            plot_losses(train_losses, val_losses, val_epochs, epoch + 1, loss_plot_file)

            file_is_empty = not os.path.exists(metrics_file) or os.path.getsize(metrics_file) == 0
            with open(metrics_file, 'a', newline='') as csvfile:
                fieldnames = ['epoch', 'time_per_epoch', 'training_loss', 'validation_loss']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                if file_is_empty:
                    writer.writeheader()

                row = {
                    'epoch': epoch + 1,
                    'time_per_epoch': epoch_time,
                    'training_loss': avg_train_loss,
                    'validation_loss': avg_val_loss
                }
                writer.writerow(row)

            print(f'Validation Loss: {avg_val_loss:.4f}')

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), f"./results/{output_folder}/best_model.pth")
                print(f"Model saved at epoch {epoch + 1} with validation loss {avg_val_loss:.4f}")

            model.train()
