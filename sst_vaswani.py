import numpy as np
import spacy
import torch
from Encoder_Vaswani import Transformer
from matplotlib import pyplot as plt
from preprocessing import fetch_data_vaswani
from torch import nn, optim
from torchinfo import torchinfo


# Evaluates the model on the given data
def evaluate(model, labels_np, data_np, mask_np, eval_batch_size):
    model.eval()
    with torch.no_grad():
        diff = 0
        for i in range(len(labels_np) // eval_batch_size):
            labels_batch = labels_np[eval_batch_size * i:eval_batch_size * (i + 1), :]
            data_batch = data_np[eval_batch_size * i:eval_batch_size * (i + 1), :, :]
            mask_batch = mask_np[eval_batch_size * i:eval_batch_size * (i + 1), :, :, :]

            labels = torch.from_numpy(labels_batch).to(device).float()
            data = torch.from_numpy(data_batch).to(device).float()
            mask = torch.from_numpy(mask_batch).to(device)
            final_out = model(data, mask)

            plug_in = (np.sign(final_out.reshape(final_out.shape[0]).cpu().detach().numpy()
                               - 0.5) + 1) / 2
            exact_labels = labels[:, 0].cpu().detach().numpy().round(decimals=0)
            diff = diff + np.sum(np.absolute((plug_in - exact_labels)))

        labels_batch = labels_np[eval_batch_size * (i + 1):, :]
        data_batch = data_np[eval_batch_size * (i + 1):, :, :]
        mask_batch = mask_np[eval_batch_size * (i + 1):, :, :, :]

        labels = torch.from_numpy(labels_batch).to(device).float()
        data = torch.from_numpy(data_batch).to(device).float()
        mask = torch.from_numpy(mask_batch).to(device)
        final_out = model(data, mask)

        plug_in = (np.sign(final_out.reshape(final_out.shape[0]).cpu().detach().numpy()
                           - 0.5) + 1) / 2
        exact_labels = labels[:, 0].cpu().detach().numpy().round(decimals=0)
        diff = diff + np.sum(np.absolute((plug_in - exact_labels)))
        acc = 1 - diff / len(labels_np)
    return acc


if __name__ == "__main__":
    np.random.seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    label_smoothing = 0.0
    epochs = 6000
    d_model = 300
    N = 2
    d_ff = d_model
    h = 2
    l_max = 56
    dropout = 0.1
    learning_rate = 0.00003
    warmup_period = 1200
    eval_ratio = 100
    eval_batch_size = 500
    error = 0.05
    nlp = spacy.load('en_core_web_lg')

    model = Transformer(d_model=d_model,
                        N=N,
                        d_ff=d_ff,
                        h=h,
                        l_max=l_max,
                        dropout=dropout
                        ).to(device)
    print(torchinfo.summary(model))

    criterion = nn.MSELoss(reduction='mean')
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.98), eps=1e-09)
    lambda1 = lambda epoch: np.minimum((epoch + 1) ** (-0.5), (epoch + 1) * warmup_period ** (-1.5))

    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)

    losses = []
    accuracies = []

    labels_train, data_train, mask_train = fetch_data_vaswani(l_max, d_model, h, nlp,
                                                              easy=False, split='train')
    labels_dev, data_dev, mask_dev = fetch_data_vaswani(l_max, d_model, h, nlp,
                                                        easy=False, split='validation')

    batch_size = 700

    indices = np.arange(len(labels_train))
    for epoch in range(epochs):
        epoch_loss = 0
        np.random.shuffle(indices)
        for i in range(len(labels_train) // batch_size):
            labels_train_batch = labels_train[indices[batch_size * i:batch_size * (i + 1)], :]
            data_train_batch = data_train[indices[batch_size * i:batch_size * (i + 1)], :, :]
            mask_train_batch = mask_train[indices[batch_size * i:batch_size * (i + 1)], :, :, :]

            labels = torch.from_numpy(labels_train_batch).to(device).float()
            data = torch.from_numpy(data_train_batch).to(device).float()
            mask = torch.from_numpy(mask_train_batch).to(device)

            out = model(data, mask)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            epoch_loss = epoch_loss + loss.item()
        epoch_loss = epoch_loss / (len(labels_train) // batch_size)
        losses.append(epoch_loss)

        lr_scheduler.step()

        if epoch % eval_ratio == 0:
            print("Epoch: " + str(epoch))
            print("Loss on train set: " + str(losses[epoch]))
            dev_acc = evaluate(model, labels_dev, data_dev, mask_dev, eval_batch_size)
            accuracies.append(dev_acc)
            print("Accuracy on dev set: " + str(dev_acc))
            print(
                "Accuracy on train set: " + str(evaluate(model, labels_train, data_train,
                                                         mask_train, eval_batch_size)))

    fig1 = plt.figure()
    ax1 = fig1.add_subplot()
    ax1.plot(np.arange(len(losses)), losses)
    ax1.plot(np.arange(len(losses)), 0.25 * np.ones(len(losses)))
    plt.title("Loss on training set")
    ax1.set_xlabel('epoch')
    ax1.set_ylabel('mean square error')
    plt.show()

    fig2 = plt.figure()
    ax2 = fig2.add_subplot()

    ax2.scatter(np.arange(0, len(losses), eval_ratio), accuracies)
    final_dev_acc = evaluate(model, labels_dev, data_dev, mask_dev, eval_batch_size)
    print("Accuracy on validation set: " + str(final_dev_acc))
    ax2.scatter([len(losses)], [final_dev_acc])
    plt.title("Accuracy on validation set")
    ax2.set_xlabel('epoch')
    ax2.set_ylabel('accuracy')
    plt.show()

    labels_test, data_test, mask_test = fetch_data_vaswani(l_max, d_model, h, nlp,
                                                           easy=False, split='test')
    print("Accuracy on test set: " + str(evaluate(model, labels_test, data_test,
                                                  mask_test, eval_batch_size)))
