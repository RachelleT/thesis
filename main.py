from collections import defaultdict
from copy import deepcopy
import random
import torch.utils
from eval import compute_metrics
from Models.dcgan import Discriminator_256, Generator_256
from Preprocessing.preprocessing import Classifier_Preprocessing, GAN_Categories, GAN_Entities
from Preprocessing.dataset import Entities_Dataset, Categories_Dataset
from Models.classifier import ResNet
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import torchvision.utils as vutils
from sklearn.model_selection import KFold
from Models import pggan
from datetime import datetime

from utils import weights_init


def training_classifier(training_data):

    batch_size = 64

    # Apply Transforms
    transform_train = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(30),
        transforms.ToTensor(),
    ])

    train_dataset = Entities_Dataset(training_data, transform_train)

    # Define the data loaders for the current fold
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
    )

    # Define the device (CPU or GPU)
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

    net = ResNet.ResNet50(10, 1).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.1,
                          momentum=0.9, weight_decay=0.0001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.1, patience=5)

    EPOCHS = 200
    for epoch in range(EPOCHS):
        losses = []
        running_loss = 0
        for i, inp in enumerate(train_loader):

            inputs, labels = inp
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            losses.append(loss.item())

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if i % 100 == 0 and i > 0:
                print(
                    f'Loss [{epoch + 1}, {i}](epoch, minibatch): ', running_loss / 100)
                running_loss = 0.0

        avg_loss = sum(losses) / len(losses)
        scheduler.step(avg_loss)

    print('Training Done')


def cross_validation_classifier(training_data):
    # Apply Transforms
    transform_train = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(30),
        transforms.ToTensor(),
    ])

    train_dataset = Entities_Dataset(training_data, transform_train)

    # Define the number of folds, batch size and loss function
    k_folds = 5
    batch_size = 64
    loss_function = nn.CrossEntropyLoss()

    # For fold results
    results = {}
    class_correct = defaultdict(int)
    class_total = defaultdict(int)

    # Set fixed random number seed
    torch.manual_seed(42)

    # Define the device (CPU or GPU)
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

    # Initialize the k-fold cross validation
    kf = KFold(n_splits=k_folds, shuffle=True)

    # Loop through each fold
    for fold, (train_idx, test_idx) in enumerate(kf.split(train_dataset)):
        print(f"Fold {fold + 1}")
        print("----------")

        # Define the data loaders for the current fold
        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
            sampler=torch.utils.data.SubsetRandomSampler(train_idx),
        )
        test_loader = DataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
            sampler=torch.utils.data.SubsetRandomSampler(test_idx),
        )

        # Initialize the model and optimizer
        model = ResNet.ResNet50(10, 1).to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.01)

        # Train the model on the current fold
        for epoch in range(0, 200):
            # Print epoch
            # print(f'Starting epoch {epoch + 1}')

            # Set current loss value
            current_loss = 0.0
            for i, data in enumerate(train_loader):

                # Get inputs
                inputs, targets = data
                inputs, targets = inputs.to(device), targets.to(device)

                # Zero the gradients
                optimizer.zero_grad()

                # Perform forward pass
                outputs = model(inputs)

                # Compute loss
                loss = loss_function(outputs, targets)

                # Perform backward pass
                loss.backward()

                # Perform optimization
                optimizer.step()

                # Print statistics
                current_loss += loss.item()
                if i % 500 == 499:
                    print('Loss after mini-batch %5d: %.3f' %
                          (i + 1, current_loss / 500))
                    current_loss = 0.0

        # Process is complete.
        print('Training process has finished. Saving trained model.')

        # Print about testing
        print('Starting testing')

        # Evaluation for this fold
        correct, total = 0, 0
        with torch.no_grad():
            for i, data in enumerate(test_loader, 0):
                # Get inputs
                inputs, targets = data
                inputs, targets = inputs.to(device), targets.to(device)

                # Generate outputs
                outputs = model(inputs)

                # Set total and correct
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()

                # Calculate intraclass accuracy
                for pred, label in zip(predicted, targets):
                    if pred == label:
                        class_correct[label.item()] += 1
                    class_total[label.item()] += 1

            # Print accuracy
            print('Accuracy for fold %d: %d %%' %
                  (fold + 1, 100.0 * correct / total))
            print('--------------------------------')
            results[fold + 1] = 100.0 * (correct / total)

    # Print fold results
    print(f'K-FOLD CROSS VALIDATION RESULTS FOR {k_folds} FOLDS')
    print('--------------------------------')
    final_sum = 0.0
    for key, value in results.items():
        print(f'Fold {key}: {value} %')
        final_sum += value
    print(f'Average: {final_sum / len(results.items())} %')

    # Calculate intraclass accuracy
    print('Intraclass Accuracy:')
    for label in class_correct:
        accuracy = 100 * class_correct[label] / class_total[label]
        print(f'Class {label}: {accuracy}%')


def training_dcgan(training_data, images_class):

    # Set random seed for reproducibility
    manualSeed = 999
    # manualSeed = random.randint(1, 10000) # use if you want new results
    print("Random Seed: ", manualSeed)
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    torch.use_deterministic_algorithms(True)  # Needed for reproducible results

    # Number of workers for dataloader
    workers = 2

    # Batch size during training
    batch_size = 32

    # Spatial size of training images. All images will be resized to this
    #   size using a transformer.
    image_size = 256

    # Number of channels in the training images. For color images this is 3
    nc = 1

    # Size of z latent vector (i.e. size of generator input)
    nz = 100

    # Size of feature maps in generator
    ngf = 8

    # Size of feature maps in discriminator
    ndf = 8

    # Number of training epochs
    num_epochs = 5

    # Learning rate for optimizers
    lr = 0.0001 

    # Beta1 hyperparameter for Adam optimizers
    beta1 = 0.5

    # Number of GPUs available. Use 0 for CPU mode.
    ngpu = 1

    # Apply Transforms
    transform_train = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5), (0.5)),
    ])

    train_dataset = Categories_Dataset(training_data, transform_train)

    # Create the dataloader
    dataloader = DataLoader(train_dataset, batch_size=batch_size,
                            shuffle=True, num_workers=workers)

    # Decide which device we want to run on
    device = torch.device("cuda:2" if (
        torch.cuda.is_available() and ngpu > 0) else "cpu")

    # Create the generator
    netG = Generator_256(ngpu, nz, ngf, nc).to(device)

    # Handle multi-GPU if desired
    if (device.type == 'cuda') and (ngpu > 1):
        netG = nn.DataParallel(netG, list(range(ngpu)))

    # Apply the ``weights_init`` function to randomly initialize all weights
    #  to ``mean=0``, ``stdev=0.02``.
    netG.apply(weights_init)

    # Print the model
    print(netG)

    # Create the Discriminator
    netD = Discriminator_256(ngpu, nc, ndf).to(device)

    # Handle multi-GPU if desired
    if (device.type == 'cuda') and (ngpu > 1):
        netD = nn.DataParallel(netD, list(range(ngpu)))

    # Apply the ``weights_init`` function to randomly initialize all weights
    # like this: ``to mean=0, stdev=0.2``.
    netD.apply(weights_init)

    # Print the model
    print(netD)

    # Initialize the ``BCELoss`` function
    criterion = nn.BCELoss()

    # Create batch of latent vectors that we will use to visualize
    #  the progression of the generator
    fixed_noise = torch.randn(64, nz, 1, 1, device=device)

    # Establish convention for real and fake labels during training
    real_label = 1.
    fake_label = 0.

    # Setup Adam optimizers for both G and D
    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

    # Lists to keep track of progress
    img_list = []
    img_list_only = []
    G_losses = []
    D_losses = []
    iters = 0

    # current time stamp
    start = datetime.now()

    print("Starting Training Loop...")
    # For each epoch
    for epoch in range(num_epochs):
        # For each batch in the dataloader
        for i, data in enumerate(dataloader, 0):
            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            # Train with all-real batch
            netD.zero_grad()
            # Format batch
            real_cpu = data[0].to(device)
            b_size = real_cpu.size(0)
            label = torch.full((b_size,), real_label,
                               dtype=torch.float, device=device)
            # Forward pass real batch through D
            output = netD(real_cpu).view(-1)
            # Calculate loss on all-real batch
            errD_real = criterion(output, label)
            # Calculate gradients for D in backward pass
            errD_real.backward()
            D_x = output.mean().item()

            # Train with all-fake batch
            # Generate batch of latent vectors
            noise = torch.randn(b_size, nz, 1, 1, device=device)
            # Generate fake image batch with G
            fake = netG(noise)
            label.fill_(fake_label)
            # Classify all fake batch with D
            output = netD(fake.detach()).view(-1)
            # Calculate D's loss on the all-fake batch
            errD_fake = criterion(output, label)
            # Calculate the gradients for this batch, accumulated (summed) with previous gradients
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            # Compute error of D as sum over the fake and the real batches
            errD = errD_real + errD_fake
            # Update D
            optimizerD.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            netG.zero_grad()
            label.fill_(real_label)  # fake labels are real for generator cost
            # Since we just updated D, perform another forward pass of all-fake batch through D
            output = netD(fake).view(-1)
            # Calculate G's loss based on this output
            errG = criterion(output, label)
            # Calculate gradients for G
            errG.backward()
            D_G_z2 = output.mean().item()
            # Update G
            optimizerG.step()

            # Output training stats
            if i % 50 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                      % (epoch, num_epochs, i, len(dataloader),
                         errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

            # Save Losses for plotting later
            G_losses.append(errG.item())
            D_losses.append(errD.item())

            # Check how the generator is doing by saving G's output on fixed_noise
            if (iters % 500 == 0) or ((epoch == num_epochs - 1) and (i == len(dataloader) - 1)):
                with torch.no_grad():
                    fake = netG(fixed_noise).detach().cpu()
                img_list.append(vutils.make_grid(fake, padding=2, normalize=True))
                torchvision.utils.save_image(vutils.make_grid(img_list[-1][0], normalize=True).cpu(), 
                                            'Results/Categories/' + images_class + '-fake/dcgan-' 
                                            + str(i+1) + '.png')
                img_list_only.append(fake)

            iters += 1

    end = datetime.now()
    td = (end - start).total_seconds() * 10**3
    print(f"The time of execution of above program is : {td:.03f}ms")

    real_batch = next(iter(dataloader))
    torchvision.utils.save_image(vutils.make_grid(real_batch[0].to(device)[:64], 
                                                  padding=5, normalize=True).cpu(), 
                                                  'Results/dcgan-real.png')
    
    grid_tensor = vutils.make_grid(fake, padding=2, normalize=True)
    grid_array = fake.numpy()
    # Extracting one image from the grid
    image_index = 0  # Change this index to extract different images
    single_image = grid_array[:, image_index * (grid_tensor.size(1) + 2): (image_index + 1) * (grid_tensor.size(1) + 2)]
    torchvision.utils.save_image(vutils.make_grid(single_image, normalize=True).cpu(), 
                                            'Results/Categories/' + images_class + '-fake/dcgan-' 
                                            + str(i+1) + '.png')

    #for i in range(0, fake.size()[0]):
    #    img_normalized = transform_norm(fake[i])
    #    torchvision.utils.save_image(img_normalized, 'Results/Categories/' + images_class + '-fake/dcgan-' + str(i+1) + '.png')

    # compute_metrics_old(real=real_batch, fakes=img_list_only, image_size)

def train_pggan(training_data):
    
    num_epochs = 50
    latent_dim = 100
    batch_size = 32

    ngpu = 1

    device = torch.device("cuda:0" if (
        torch.cuda.is_available() and ngpu > 0) else "cpu")
    
    generator = pggan.Generator(latent_dim).to(device)
    discriminator = pggan.Discriminator().to(device)

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_dataset = Categories_Dataset(training_data, transform)
    
    # Create the dataloader
    dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                            num_workers=2)

    generator = pggan.Generator(latent_dim).to(device)
    discriminator = pggan.Discriminator().to(device)

    criterion = nn.BCELoss()
    optimizer_G = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

    real_label = 1
    fake_label = 0

    for epoch in range(num_epochs):
        for i, data in enumerate(dataloader):
            real_images, _ = data
            batch_size = real_images.size(0)

            # Train Discriminator
            discriminator.zero_grad()
            real_images = real_images.to(device)
            output = discriminator(real_images, 4)
            label = torch.full((batch_size,), real_label, device=device)
            errD_real = criterion(output.view(-1), label)
            errD_real.backward()
            D_x = output.mean().item()

            noise = torch.randn(batch_size, latent_dim, 1, 1, device=device)
            fake_images = generator(noise, 4)
            output = discriminator(fake_images.detach(), 4)
            label.fill_(fake_label)
            errD_fake = criterion(output.view(-1), label)
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            errD = errD_real + errD_fake
            optimizer_D.step()

            # Train Generator
            generator.zero_grad()
            label.fill_(real_label)
            output = discriminator(fake_images, 4)
            errG = criterion(output.view(-1), label)
            errG.backward()
            D_G_z2 = output.mean().item()
            optimizer_G.step()

            print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
                  % (epoch, num_epochs, i, len(dataloader),
                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

            if i % 100 == 0:
                torch.utils.save_image(fake_images[:25], 'generated_images_epoch_%d_%d.png' % 
                                       (epoch, i), nrow=5, normalize=True)


if __name__ == '__main__':

    # Preprocessing - Classifier
    raw_data = Classifier_Preprocessing()
    train, test = raw_data.split_data()

    # Train Classifier
    # cross_validation_classifier(train)
    # training_classifier(train)

    # Pretraining - GAN
    entities_classes = ["aneurysmatic bone cyst", "chondroblastoma", "chondrosarcoma",
               "enchondroma", "ewing sarcoma", "fibruous dysplasia",
               "giant cell tumour", "non-ossifying fibroma", "osteochondroma",
               "osteosarcoma"]
    gan_data = GAN_Entities(entities_classes[8])
    train_gan = gan_data.class_data()

    # Categories

    categories_classes = ["benign", "intermediate", "malignant"]

    category_data = GAN_Categories(categories_classes[0])
    train_category = category_data.class_data()

    # Train DCGAN
    training_dcgan(train_category, categories_classes[0])

    # Train PGGAN
    # train_pggan(train_category)
