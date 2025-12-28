import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import torch
from sklearn.model_selection import train_test_split
import torch.nn as nn
from transformers import ViTModel, ViTForImageClassification, ViTImageProcessor
import torch.optim as optim
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from zipfile import ZipFile

if not os.path.exists("labels.csv"):
	dataset = pd.read_csv('list_attr_celeba.txt', sep='\s+', skiprows=1, nrows=202599)
	labels = dataset["Male"]
	labels.to_csv("labels.csv", index=True)
 
data = pd.read_csv("labels.csv", index_col=0)

df_train, df_temp = train_test_split(data, test_size=0.2, stratify=data['Male'], random_state=42)
df_eval, df_test = train_test_split(df_temp, test_size=0.5, stratify=df_temp['Male'], random_state=42)

class ImageLabelDataset(Dataset):
	def __init__(self, data, img_dir, transform=None):
		self.data = data
		self.img_dir = img_dir
		self.transform = transform

	def __len__(self):
		return len(self.data)

	def __getitem__(self, idx):
		row = self.data.iloc[idx]
		img_path = os.path.join(self.img_dir, row.name)
		try:
			image = Image.open(img_path).convert("RGB")
		except Exception as e:
			print(f"[ERROR] Failed to load image: {img_path} - {e}")
			image = Image.new("RGB", (224, 224)) 
		label = int(row['Male'])
		label = int((label + 1) // 2)
		if self.transform:
			try:
				image = self.transform(image)
			except Exception as e:
				print(f"[ERROR] Transform failed for {img_path} - {e}")
				image = torch.zeros(3, 224, 224)

		return image, label
# Configuration
MODEL_NAME = 'google/vit-base-patch16-224'
NUM_CLASSES = 2  # Binary classification (e.g., gender)
BATCH_SIZE = 128
EPOCHS = 50
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
criterion = nn.CrossEntropyLoss()
# === Define Model ===
class GenderViT(nn.Module):
	def __init__(self):
		super().__init__()
		self.model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
		self.model.eval()
		for param in self.model.parameters():
			param.requires_grad = False  # Freeze backbone

		hidden_dim = self.model.config.hidden_size
		self.model.classifier = nn.Linear(hidden_dim, NUM_CLASSES)

	def forward(self, x):
		with torch.no_grad():  # Backbone is frozen
			outputs = self.model(x)
		return outputs.logits  # Return logits for classification

def evaluate(model, dataloader, device):
	model.eval()
	correct = 0
	total = 0
	total_loss = 0.0

	with torch.no_grad():
		for images, labels in dataloader:
			images, labels = images.to(device), labels.to(device)
			outputs = model(images)
			outputs = outputs.logits if hasattr(outputs, 'logits') else outputs  # Handle different model outputs
			loss = criterion(outputs, labels)
			total_loss += loss.item()

			_, predicted = outputs.max(1)
			correct += predicted.eq(labels).sum().item()
			total += labels.size(0)

	accuracy = correct / total * 100
	avg_loss = total_loss / len(dataloader)
	return accuracy, avg_loss


def main():

	writer = SummaryWriter(log_dir="runs/vit_gender_cls_single")
	PATIENCE = 5        # Number of epochs to wait for improvement
	BEST_ACC = 0.0
	EPOCHS_NO_IMPROVE = 0
	NUM_WORKERS = 8

	transform = transforms.Compose([
		transforms.RandomResizedCrop(224, scale=(0.8, 1.0), ratio=(0.9, 1.1)),  # crop & resize
		transforms.RandomHorizontalFlip(p=0.5),  # natural for face data
		transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),  # light color jitter
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.5]*3, std=[0.5]*3),  # match ViT pretraining
	])
	eval_transform = transforms.Compose([
		transforms.Resize(256),
		transforms.CenterCrop(224),
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.5]*3, std=[0.5]*3),
	])


	# Replace this with your real gender dataset
	train_dataset = ImageLabelDataset(df_train, "img_align_celeba/img_align_celeba", transform=transform)
	eval_dataset = ImageLabelDataset(df_eval, "img_align_celeba/img_align_celeba", transform=eval_transform)
	test_dataset = ImageLabelDataset(df_test, "img_align_celeba/img_align_celeba", transform=eval_transform)
	train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, persistent_workers=False, pin_memory=True )
	eval_loader = DataLoader(eval_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, persistent_workers=False,pin_memory=True )
	test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, persistent_workers=False,pin_memory=True )

	# === Training ===
	model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
	for param in model.parameters():
		param.requires_grad = False  
	hidden_dim = model.config.hidden_size
	model.classifier = nn.Linear(hidden_dim, NUM_CLASSES)
	criterion = nn.CrossEntropyLoss()
	optimizer = optim.AdamW(model.classifier.parameters(), lr=3e-4, weight_decay=1e-4)
	model.to(DEVICE)


	for epoch in range(EPOCHS):
		model.train()
		running_loss = 0.0
		correct = 0
		total = 0

		for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
			images, labels = images.to(DEVICE), labels.to(DEVICE)

			optimizer.zero_grad()
			outputs = model(images)
			outputs = outputs.logits if hasattr(outputs, 'logits') else outputs  # Handle different model outputs
			loss = criterion(outputs, labels)
			loss.backward()
			optimizer.step()

			running_loss += loss.item()
			_, predicted = outputs.max(1)
			correct += predicted.eq(labels).sum().item()
			total += labels.size(0)

		train_acc = correct / total * 100
		train_loss = running_loss / len(train_loader)

		# 🔍 Evaluate on validation set
		eval_acc, eval_loss = evaluate(model, eval_loader, DEVICE)
		# Log metrics to TensorBoard
		writer.add_scalar("Loss/Train", train_loss, epoch)
		writer.add_scalar("Accuracy/Train", train_acc, epoch)
		writer.add_scalar("Loss/Eval", eval_loss, epoch)
		writer.add_scalar("Accuracy/Eval", eval_acc, epoch)

		print(f"Epoch [{epoch+1}/{EPOCHS}] Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | Eval Acc: {eval_acc:.2f}% | Eval Loss: {eval_loss:.4f}")
		# Early stopping logic
		if eval_acc > BEST_ACC:
			BEST_ACC = eval_acc
			EPOCHS_NO_IMPROVE = 0
			# Save model checkpoint
			torch.save(model.state_dict(), f"best_model_{epoch}.pth")
		else:
			EPOCHS_NO_IMPROVE += 1
			print(f"No improvement for {EPOCHS_NO_IMPROVE} epoch(s)")

		if EPOCHS_NO_IMPROVE >= PATIENCE:
			print(f"Early stopping at epoch {epoch+1}. Best Eval Acc: {BEST_ACC:.2f}%")
			break
		writer.add_scalar("Best/EvalAccuracy", BEST_ACC, epoch)
  
	test_acc, test_loss = evaluate(model, test_loader, DEVICE)
	print(f"\nFinal Test Accuracy: {test_acc:.2f}% | Test Loss: {test_loss:.4f}")
	writer.add_scalar("Accuracy/Test", test_acc, EPOCHS)
	writer.add_scalar("Loss/Test", test_loss, EPOCHS)
	writer.close()

	# Save the model
	torch.save(model.state_dict(), "vit_gender_cls_single.pth")
 
if __name__ == "__main__":
	import multiprocessing
	multiprocessing.set_start_method('spawn', force=True)
	main()
