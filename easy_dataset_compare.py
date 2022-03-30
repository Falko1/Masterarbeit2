import time
import numpy as np
import spacy
import torch
from Encoder_Kohler import Transformer_Kohler
from Encoder_Vaswani import Transformer
from matplotlib import pyplot as plt
from preprocessing import fetch_data_kohler, fetch_data_vaswani
from torch import nn, optim
from torchinfo import torchinfo


# Trains the given model with the hyperparameters for epochs.
def do_train(trafo_variant, epochs, N, h, I, d, l_max, learning_rate, warmup_period, eval_ratio):
    print(f"Training ({trafo_variant})")
    np.random.seed(random_state)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    nlp = spacy.load('en_core_web_lg')

    assert trafo_variant in ['vaswani', 'kohler'], 'model name not supported!'
    if trafo_variant == 'vaswani':
        d_model = d
        d_ff = d_model
        model = Transformer(d_model=d_model,
                            N=N,
                            d_ff=d_ff,
                            h=h,
                            l_max=l_max,
                            dropout=0
                            ).to(device)
        labels_train, data_train, mask_train = fetch_data_vaswani(l_max, d_model, h, nlp,
                                                                  easy=True, split='train')
        labels_dev, data_dev, mask_dev = fetch_data_vaswani(l_max, d_model, h, nlp,
                                                            easy=True, split='dev')
        labels_test, data_test, mask_test = fetch_data_vaswani(l_max, d_model, h, nlp,
                                                               easy=True, split='test')
    else:
        d_model = (d + l_max + 4) * h * I
        d_k = d_model // h
        d_ff = d_model
        model = Transformer_Kohler(d_model=d_model,
                                   N=N,
                                   d_k=d_k,
                                   d_ff=d_ff,
                                   h=h,
                                   l_max=l_max,
                                   dropout=0
                                   ).to(device)
        labels_train, data_train, mask_train = fetch_data_kohler(l_max, d, h, I, nlp,
                                                                 easy=True, split='train')
        labels_dev, data_dev, mask_dev = fetch_data_kohler(l_max, d, h, I, nlp,
                                                           easy=True, split='dev')
        labels_test, data_test, mask_test = fetch_data_kohler(l_max, d, h, I, nlp,
                                                              easy=True, split='test')

    print(torchinfo.summary(model))

    criterion = nn.MSELoss(reduction='mean')
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.98), eps=1e-09)
    lambda1 = lambda epoch: np.minimum((epoch + 1) ** (-0.5), (epoch + 1) * warmup_period ** (-1.5))

    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)

    losses = []
    accuracies = []

    batch_size = len(labels_train)
    t0 = time.time()
    indices = np.arange(batch_size)
    for epoch in range(epochs):

        np.random.shuffle(indices)

        labels_train_batch = labels_train[indices[:batch_size], :]
        data_train_batch = data_train[indices[: batch_size], :, :]
        mask_train_batch = mask_train[indices[: batch_size], :, :, :]

        labels = torch.from_numpy(labels_train_batch).to(device).float()
        data = torch.from_numpy(data_train_batch).to(device).float()
        mask = torch.from_numpy(mask_train_batch).to(device)

        out = model(data, mask)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        epoch_loss = loss.item()

        losses.append(epoch_loss)

        lr_scheduler.step()
        if epoch % eval_ratio == 0:
            print(f"Epoch ({trafo_variant}): " + str(epoch))
            print(f"Loss on train set ({trafo_variant}): " + str(losses[epoch]))
            dev_acc = evaluate(model, labels_dev, data_dev, mask_dev, device)
            accuracies.append(dev_acc)
            print(f"Accuracy on dev set ({trafo_variant}): " + str(dev_acc))
            print(f"Accuracy on train set ({trafo_variant}): " + str(
                evaluate(model, labels_train, data_train, mask_train, device)))
        if epoch == 1000:
            print("half-time test accuracy: " + str(evaluate(model, labels_test, data_test,
                                                             mask_test, device)))

    t1 = time.time()
    final_acc_dev = evaluate(model, labels_dev, data_dev, mask_dev, device)
    final_acc_test = evaluate(model, labels_test, data_test, mask_test, device)
    return [losses, accuracies, final_acc_dev, final_acc_test, t1 - t0]


# Evaluate the model on the given data
def evaluate(model, labels, data, mask, device):
    model.eval()
    with torch.no_grad():
        labels = torch.from_numpy(labels).to(device).float()
        data = torch.from_numpy(data).to(device).float()
        mask = torch.from_numpy(mask).to(device)
        final_out = model(data, mask)
        plug_in = (np.sign(final_out.reshape(final_out.shape[0]).cpu().detach().numpy()
                           - 0.5) + 1) / 2
        exact_labels = labels[:, 0].cpu().detach().numpy().round(decimals=0)
        diff = np.absolute((plug_in - exact_labels))
        acc = 1 - np.sum(diff) / len(labels)
    return acc


if __name__ == "__main__":
    random_state = 42
    epochs = 2000
    d = 300
    N = 2
    h = 2
    I = 1
    l_max = 6
    learning_rate = 0.00003
    warmup_period = 400
    eval_ratio = 100
    error = 0.05

    result_vaswani = do_train('vaswani', epochs, N, h, I, d, l_max, learning_rate,
                              warmup_period, eval_ratio)
    result_kohler = do_train('kohler', epochs, N, h, I, d, l_max, learning_rate,
                             warmup_period, eval_ratio)

    no_evals = len(result_vaswani[0])

    fig1 = plt.figure()
    ax1 = fig1.add_subplot()
    ax1.plot(np.arange(no_evals), result_vaswani[0], color='blue', label='Vaswani')
    ax1.plot(np.arange(no_evals), result_kohler[0], color='orange', label='Kohler')
    ax1.plot(np.arange(no_evals), 0.25 * np.ones(no_evals), color='black', linestyle="dashed")
    plt.title("Loss on training set")
    plt.legend()
    ax1.set_xlabel('epoch')
    ax1.set_ylabel('mean square error')
    plt.show()

    fig2 = plt.figure()
    ax2 = fig2.add_subplot()
    ax2.scatter(np.arange(0, no_evals, eval_ratio), result_vaswani[1], color='blue',
                label='Vaswani', marker='*')
    ax2.scatter(np.arange(0, no_evals, eval_ratio), result_kohler[1],
                label='Kohler', marker='o',
                facecolors='none', edgecolors='orange')
    plt.title("Accuracy on validation set")
    plt.legend()
    ax2.set_xlabel('epoch')
    ax2.set_ylabel('accuracy')
    plt.show()

    print('Vaswani result on test set: ' + str(result_vaswani[3]))
    print('Kohler result on test set: ' + str(result_kohler[3]))
    print('Vaswani time taken: ' + str(result_vaswani[4]))
    print('Kohler time taken: ' + str(result_kohler[4]))
