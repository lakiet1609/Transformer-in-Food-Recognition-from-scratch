import torch
import torchvision
import dataloader, engine, utils_file, predictions
from tranformer_model import transformer_vision_model
from torchmetrics import Accuracy
from timeit import default_timer as timer
import random
from pathlib import Path

def main():
    # Setup hyperparameters
    image_size = 224
    channels = 3 #This mean 3 channel of RGB images 
    epochs = 30
    batch_size = 32
    learning_rate = 0.001
    start_time = timer()

  # Setup directories
    train_dir = "data/Food/train"
    test_dir = "data/Food/test"

    # Setup target device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Create DataLoaders with help from data_setup.py
    train_dataloader, test_dataloader, class_names = dataloader.create_dataloaders(
        train_dir=train_dir,
        test_dir=test_dir,
        batch_size=batch_size
    )

  # Set the manual seeds
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    output_shape = len(class_names)

    #Get a vision transform model
    vision_transformer = transformer_vision_model(input_channel = channels, 
                                                  image_size = image_size, 
                                                  num_classes = output_shape)

    
    # Set loss, optimizer and accuracy
    accuracy= Accuracy(task='multiclass', num_classes=output_shape).to(device)
    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(vision_transformer.parameters(),
                                 lr=learning_rate,
                                 weight_decay=0.3
                                 )

    # Start training with help from engine.py
    results = engine.training(model=vision_transformer,
              train_dataloader=train_dataloader,
              test_dataloader=test_dataloader,
              loss_function=loss_function,
              optimizer=optimizer,
              accuracy = accuracy,
              epochs=epochs,
              device=device)

    end_time = timer()
    print(f"[INFO] Total training time: {end_time-start_time:.3f} seconds")

    #Model Plotting
    utils_file.plot_evaluation_curve(results)

    # Get a random list of image paths from test set
    num_images_to_plot = 3
    test_image_path_list = list(Path(test_dir).glob("*/*.jpg")) 
    test_image_path_sample = random.sample(population=test_image_path_list, 
                                        k=num_images_to_plot) 

    # Make predictions on and plot the images
    for image_path in test_image_path_sample:
        predictions.pred_and_plot_image(model=vision_transformer, 
                                      image_path=image_path,
                                      class_names=class_names,
                                      image_size=(224, 224))

    utils_file.save_model(model=vision_transformer,
                          target_dir='models',
                          model_name='model_0.pth')

    
if __name__ == '__main__':
  main()


    
    

