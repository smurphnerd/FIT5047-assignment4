import torch


class BaseTrainer:
    def __init__(self, model, criterion, optimizer, train_loader, val_loader, device):
        self.model = model.to(device)
        self.criterion = criterion  # the loss function
        self.optimizer = optimizer  # the optimizer
        self.train_loader = train_loader  # the train loader
        self.val_loader = val_loader  # the valid loader
        self.device = device

    # the function to train the model in many epochs
    def fit(self, num_epochs):
        self.num_batches = len(self.train_loader)

        for epoch in range(num_epochs):
            print(f"Epoch {epoch + 1}/{num_epochs}")
            train_loss, train_accuracy = self.train_one_epoch()
            val_loss, val_accuracy = self.validate_one_epoch()
            # print(
            #     f"{self.num_batches}/{self.num_batches} - train_loss: {train_loss:.4f} - train_accuracy: {train_accuracy*100:.4f}%"
            # )
            print(
                f"{self.num_batches}/{self.num_batches} - train_loss: {train_loss:.4f} - train_accuracy: {train_accuracy*100:.4f}% \
                - val_loss: {val_loss:.4f} - val_accuracy: {val_accuracy*100:.4f}%"
            )

    # train in one epoch, return the train_acc, train_loss
    def train_one_epoch(self):
        self.model.train()
        running_loss, correct, total = 0.0, 0, 0
        for i, data in enumerate(self.train_loader):
            inputs, labels = data
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        train_accuracy = correct / total
        train_loss = running_loss / self.num_batches
        return train_loss, train_accuracy

    # evaluate on a loader and return the loss and accuracy
    def evaluate(self, loader):
        self.model.eval()
        loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for data in loader:
                inputs, labels = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = correct / total
        loss = loss / len(self.val_loader)
        return loss, accuracy

    def evaluate_dual_labels(self, loader):
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for data in loader:
                (
                    inputs,
                    dual_labels,
                ) = data  # dual_labels contains lists of multiple possible labels per input
                inputs = inputs.to(self.device)
                dual_labels = [label_set.to(self.device) for label_set in dual_labels]

                outputs = self.model(inputs)
                loss = 0.0

                # Compute loss for each sample based on the multiple labels
                for i, label_set in enumerate(dual_labels):
                    # Compute the loss against each of the possible labels
                    per_sample_loss = torch.mean(
                        torch.stack(
                            [self.criterion(outputs[i], label) for label in label_set]
                        )
                    )
                    loss += per_sample_loss.item()

                total_loss += loss

                # Get predictions
                _, predicted = torch.max(outputs.data, 1)

                # For each sample, check if the predicted label is in the set of dual labels
                for i, label_set in enumerate(dual_labels):
                    if predicted[i].item() in label_set.tolist():
                        correct += 1
                    total += 1

        # Compute overall loss and accuracy
        avg_loss = total_loss / len(loader)
        accuracy = correct / total

        return avg_loss, accuracy

    # return the val_acc, val_loss, be called at the end of each epoch
    def validate_one_epoch(self):
        val_loss, val_accuracy = self.evaluate(self.val_loader)
        return val_loss, val_accuracy
