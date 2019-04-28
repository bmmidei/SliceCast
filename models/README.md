## Overview of models in this directory:
Each time you save new weights to this directory, please rename it using the 
following format and provide some information about the model structure:
MM_DD_YYYY_time - example 04_14_2019_2000.h5

## Model History:

* 04_14_2019_1500.h5 - 2 stacked Bidirectional LSTMs with output 256 each.
Followed by dropout, then dense128 with relu, then dropout, then dense1 with
sigmoid to binary crossentropy loss.
Trained for 10 epochs, batch size 16, steps per epoch 100

* 04_14_2019_2000.h5 - 2 stacked Bidirectional LSTMs with output 256 each.
Followed by dropout, then dense256 relu, then dropout, then dense64 relu,
then droput then dense3 softmax. Categorical crossentropy loss.
Trained for 10 epochs, batch size 8, steps per epoch 500

* 04_18_2019_1600.h5 - 2 stacked Bidirectional LSTMs with output 256 each.
Followed by dropout, then dense256 relu, then dropout, then dense64 relu,
then droput then dense3 softmax. Categorical crossentropy loss with 3 classes.
Trained for 10 epochs, batch size 8, steps per epoch 1000. Trained on wiki
dataset before we removed intro paragraphs and documents with less than 3 segs.

* 04_20_2019_2300.h5 - 2 stacked Bidirectional LSTMs with output 256 each.
Followed by dropout, then dense256 relu, then dropout, then dense64 relu, 
then dense3 softmax. Categorical crossentropy loss with 3 classes. 
Trained for 20 epochs, batch size 8, steps per epoch 1000. Trained on "train" wiki
dataset after we removed intro paragraphs and documents with less than 3 segs.
Class weights = [1.0, 10.0, 0.2]

* 04_22_2019_2200.h5 - 2 stacked Bidirectional LSTMs with output 256 each.
Followed by dropout, then dense256 relu, then dropout, then dense64 relu, 
then dense3 softmax. Categorical crossentropy loss with 3 classes. 
Trained for 20 epochs, batch size 8, steps per epoch 1000. Trained on "train" wiki
dataset after we removed intro paragraphs and documents with less than 3 segs.
Class weights = [1.0, 7.0, 0.2] => This yielded all 0 predictions

* 04_22_2019_2300_podcast - Same structure as above. Used 04_20_2019_2300 as a 
pretrained model, then trained an additional 10 epochs over 30 podcasts using
class weights of [1.0, 9.0, 0.2]

* 04_26_2019_0900_podcast - Used 04_20_2019_2300 as a pretrained model, then trained
on an additional 10 epochs over podcasts using class weights of [1.0, 20.0, 0.2].

* 04_26_2019_1000_podcast - Used 04_20_2019_2300 as a pretrained model, then trained
on an additional 10 epochs over podcasts using class weights of [1.0, 30.0, 0.2].
    We actually had pretty good wiki results with this as well.
    
    

