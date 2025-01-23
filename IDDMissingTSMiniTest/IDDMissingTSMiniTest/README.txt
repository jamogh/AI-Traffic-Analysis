Task1 output should be as follows:
For each image "n.png" there should be an output text file with name "n.txt" having predicted class followed by box in YOLO format and a prediction score.

E.g. 1.txt with following 1 line as content:-
side-road-left, 0.32, 0.45, 0.12, 0.4, 0.75

where  0.32, 0.45, 0.12, 0.4 is YOLO box, and  0.75 is prediction score.

Important Note: only one topmost box will be considered in each file during evaluation, so participants should ensure one box (and one line) per text file.
mAP (mean Average Precision) will be used as the evaluation metric.


Task 2 output should be as follows:
For each image "n.png" there should be an output text file with name "n.txt" with the predicted class and prediction score.
E.g. 1.txt with following 1 line as content:-
side-road-left, 0.79

Important Note: only one topmost class will be considered in each file during evaluation, so participants should ensure one class (and one line) per text file.
Top-1 accuracy (fraction of test images with correct predictions) will be considered for the evaluation of task2.

Make sure traffic sign class names in above two files should follow same names given in the training samples.

Final score will be calculated as 0.4 * (mAP for task 1) + 0.6 * (top-1 accuracy  for task 2). In case of conflict, priority will be given to task 2.

Create two folders with names "task1preds" and "task2preds" containing text files with predictions for task 1 and task 2 respectively. Zip them and send the mail to rohit@iitmandi.ac.in before 15 June'23 11:55 p.m.
