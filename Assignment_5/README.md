Be careful, you might encure a problem with the dataset. 

using this function call 

  train_dataset = datasets.VOCDetection(image_root, year='2007', image_set=split,
                                    download=True) 
                                    
at the time this is committed, downloaded the test set part of the dataset. I have to edit the pytorch module VOCDetection in order to download the training part. 
I will raise an issue to torchvision.datasets maintainers to fix that. 
