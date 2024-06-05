from collections import defaultdict
from copy import deepcopy
import random
import torch.utils
from eval import psnr
from Models.dcgan import Discriminator_256, Generator_256
from Preprocessing.preprocessing import Classifier_Categories, GAN_Categories, GAN_Entities
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
import matplotlib.pyplot as plt
from utils import concat_image, extract_images_from_grid, resize_image, save_image, select_random, weights_init


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
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(30),
        transforms.ToTensor(),
    ])

    train_dataset = Categories_Dataset(training_data, transform_train)

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


def train_dcgan(training_data=None, images_class=None):

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
    # batch_size = 32 -> benign
    # batch_size = 8 -> intermediate
    # batch_size = 16 -> intermediate
    batch_size = 32

    # Spatial size of training images. All images will be resized to this
    #   size using a transformer.
    # image_size = 256
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
    # num_epochs = 2500 -> benign
    # num_epochs = 800 -> intermediate
    # num_epochs = 1500 -> malignant
    num_epochs = 2500

    # Learning rate for optimizers
    # lr = 0.0001 -> benign, malignant 
    # lr = 0.001 -> intermediate
    lr = 0.0001

    # Beta1 hyperparameter for Adam optimizers
    beta1 = 0.5

    # Number of GPUs available. Use 0 for CPU mode.
    ngpu = 1

    # Apply Transforms
    transform_train = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Grayscale(),
        transforms.Resize(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5), (0.5)),
    ])

    train_dataset = Categories_Dataset(training_data, transform_train)

    # Create the dataloader
    dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=workers)

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
            # if (iters % 100 == 0) or ((epoch == num_epochs - 1) and (i == len(dataloader) - 1)):
            if (epoch + 1)  % 100 == 0:
                with torch.no_grad():
                    fake = netG(fixed_noise).detach().cpu()
                grid = vutils.make_grid(fake, padding=2, normalize=True)
                extract_images_from_grid(grid, epoch+1, 'Results/DCGAN_Categories/' + images_class)
            iters += 1

    end = datetime.now()
    td = (end - start).total_seconds() / 60
    print(f"The time of execution of above program is: {td:.03f} minutes.")

    torchvision.utils.save_image(grid, 'Results/dcgan-fake-image-norm-' + images_class + '.png')

    # Plot the loss curves
    plt.figure(figsize=(10, 5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_losses, label="G")
    plt.plot(D_losses, label="D")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig('Results/Loss_Curves/dcgan-' + images_class + '-loss-curve.png')  # Save the plot

def train_pggan(training_data=None, image_class=None):

    num_stages = 5
    num_epochs = 350 # 600 for benign
    base_channels = 8
    batch_size = [8, 8, 8, 8, 8, 8] # 32 for benign
    image_channels = 1
    ngpu = 1

    device = torch.device("cuda:2" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

    generator = pggan.Generator(max_stage=num_stages, base_channels=base_channels, image_channels=image_channels).to(device)
    discriminator = pggan.Discriminator(max_stage=num_stages, base_channels=base_channels, image_channels=image_channels).to(device)

    print(generator)

    g_optimizer = optim.Adam(generator.parameters(), lr=1e-4, betas=(0, 0.99))
    d_optimizer = optim.Adam(discriminator.parameters(), lr=1e-4, betas=(0, 0.99))

    # current time stamp
    start = datetime.now()

    for stage in range(num_stages + 1):

        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Grayscale(),
            transforms.Resize(4 * 2 ** min(stage, num_stages)),
            # transforms.Resize((4 * 2 ** min(stage, num_stages), 4 * 2 ** min(stage, num_stages))),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])])
        
        train_dataset = Categories_Dataset(training_data, transform)
        # train_dataset = MURADataset(annotations_file='/home/rachelle_tr/Documents/MISBTC/MURA-v1.1/shoulder_image_paths.csv', 
        #                            img_dir='/home/rachelle_tr/Documents/MISBTC', 
        #                            transform=transform)
    
        # Create the dataloader
        train_loader = DataLoader(train_dataset, batch_size=batch_size[stage], shuffle=True, drop_last=True)

        gp = pggan.GradientPenalty(batch_size[stage], 10, device)
        progress = pggan.Progress(num_stages, num_epochs, len(train_loader))

        generator.train()
        discriminator.train()
        for epoch in range(num_epochs):
            for i, (real_images,  label) in enumerate(train_loader):
                real_images = real_images.to(device)
                progress.progress(stage, epoch, i)

                # discriminator
                generator.zero_grad()
                discriminator.zero_grad()

                latent_dim = min(base_channels * 2 ** num_stages, 512)
                z = torch.randn(batch_size[stage], latent_dim, 1, 1, device=device)

                with torch.no_grad():
                    fake_images = generator(z, progress.alpha, progress.stage)

                d_real = discriminator(real_images, progress.alpha, progress.stage).mean()
                d_fake = discriminator(fake_images, progress.alpha, progress.stage).mean()

                gradient_penalty = gp(discriminator, real_images.data, fake_images.data, progress)

                epsilon_penalty = (d_real ** 2).mean() * 0.001

                d_loss = d_fake - d_real
                d_loss_gp = d_loss + gradient_penalty + epsilon_penalty

                d_loss_gp.backward()
                d_optimizer.step()

                # generator
                generator.zero_grad()
                discriminator.zero_grad()

                latent_dim = min(base_channels * 2 ** num_stages, 512)
                z = torch.randn(batch_size[stage], latent_dim, 1, 1, device=device)

                fake_images = generator(z, progress.alpha, progress.stage)
                g_fake = discriminator(fake_images, progress.alpha, progress.stage).mean()
                g_loss = -g_fake

                g_loss.backward()
                g_optimizer.step()

            if epoch % 100 == 0:
                print("Stage:{:>2} | Epoch :{:>3} | D_Loss:{:>10.5f} | G_Loss:{:>10.5f}".format(stage, epoch, d_loss, g_loss))
                synthesized_images = fake_images
                fake_images = fake_images.permute(0, 2, 3, 1).cpu().detach().numpy()
                generated_images = fake_images
                fake_images = concat_image(fake_images)
                fake_images = resize_image(fake_images, 224)
                save_image("Results/{}_stage_{}_epoch.jpg".format(stage, epoch), fake_images)

    print(f"GPU used {torch.cuda.memory_allocated(device) / 1024**3} GB of memory.")

    end = datetime.now()
    td = (end - start).total_seconds() / 60
    print(f"The time of execution of above program is: {td:.03f} minutes.")

    for i in range(0, batch_size[0]):
        pggan_image = resize_image(generated_images[i], 224)
        save_image('Results/PGGAN_Categories/' + image_class + '/pggan-' + str(i+1) + '.png', pggan_image) 

    real_batch = next(iter(train_loader))
    psnr_score = psnr(real_batch[0].numpy(), synthesized_images.cpu().detach().numpy())
    print("PSNR: ", psnr_score)

if __name__ == '__main__':

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
    # train_dcgan(train_category, categories_classes[0])
    # train_dcgan(training_data=None, images_class="Mura")

    # Train PGGAN
    # train_pggan(train_category, categories_classes[2])
    # train_pggan(training_data=None, image_class="Mura")

    # Preprocessing - Classifier
    real_data = Classifier_Categories("Categories")
    train, test = real_data.split_data()

    synthetic_data = Classifier_Categories("Results/DCGAN_Categories")
    synthetic_train = synthetic_data.class_data()
    
    synthetic_list = select_random(synthetic_train, 0.10)
    
    real_syn_data = train + synthetic_list

    # Train Classifier
    # cross_validation_classifier(real_syn_data)
    # training_classifier(train)
