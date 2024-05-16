The datasets given in the directory as follows:

Delhi Google Earth Folder contains two folders with images and their corrosponding Ground Truth masks.

Zurich Dataset folder contains images and the corrsponding masks.


Here, Basic steps have been given:
	1. We load the respective train and test data including their masks all in RGB and dropped the NIR band.
	2. We then Normalize each channels of each images in all train and Test images in the range 0-1.
	3. Further we extracted 224*224 patches from each train set images and created the final training set 	which is later splitted into train and valid set in the ratio 90:10. 
	4. We have use the sum of Dice loss and Focal loss as the total loss for the model. Mean IOU is used as 	the metric.
	5. For testing we split the image to 224*224 and make prediction on each of them and then stitch them in 	order.
	6. Final we compute the Mean IOU on the predicted mask and groundtruth.


## Author
1. Akash Kumar Singh
2023AIY7582 (School of Artificial Intelligence, IIT Delhi)