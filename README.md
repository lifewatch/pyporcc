# PyPorCC

Adapted code to python from the paper: 
> Cosentino, M., Guarato, F., Tougaard, J., Nairn, D., Jackson, J. C., & Windmill, J. F. C. (2019). 
> Porpoise click classifier (PorCC): A high-accuracy classifier to study harbour porpoises ( Phocoena phocoena ) in the wild . 
> The Journal of the Acoustical Society of America, 145(6), 3427–3434. https://doi.org/10.1121/1.5110908


## INFORMATION ABOUT PYTHON FILES
- click_detector.py: read sound files and create click cuttings out of it
- porcc: model class and PorCC algorithm class
- propoise_classifier: other models that can be trained to classify clicks (not PorCC)

### Examples:
- detect_clicks: how to use the click detector algorithm 
- create_model: how to train and save models from validated data. Examples for PorCC and other are provided
- classify: how to use the classifiers if the models are already trained and stored



## INFORMATION ABOUT THE DATA FILES CONTENT

trainHQ_data.mat = Matlab structure with data to to develop the logistic regression model to classify high-quality harbour porpoise clicks (HQ) 
Fields:
- *id*: identification number
- *date*: date and time when it was recorded in string format 
- *wave*: signals as recorded in the first hydrophones it impinges 
- *P*: whether the click was a HQ porpoise click (value 1) or high-frequency noise (value 0). 


trainLQ_data.mat = Matlab structure with data to develop the logistic regression model to classify low-quality harbour porpoise clicks (LQ). 
Fields:
- *id*: identification number
- *date*: date and time when it was recorded in string format 
- *wave*: signals as recorded in the first hydrophones it impinges 
- *P*: whether the click was a LQ porpoise click (value 1) or high-frequency noise (value 0)


test_data.mat = Matlab structure with data to test the result of PorCC.
Fields: 
- *id*: identification number
- *date*: date and time when it was recorded in string format 
- *realdate*: date as dd-mmm-yyyy hh:mm:ss 
- *startSample*: sample where the signal begins 
- *wave*: signals as recorded in the first hydrophones it impinges 
- *duration*: duration estimated as the 80% of the energy of the signal 
- *CF*: centroid frequency 
- *BW*: the -3dB bandwidth
- *ratio*: ratio between the peak and centroid frequency
- *XC*: maximum value of the cross-correlation coefficient carried out against a typical porpoise click, 
- *Q*: defined as the RMS bandwidth divided the centroid frequency
- *ManualAsign*: class to which the signals were manually assigned
- *ClassifiedAs*: class assigned by PorCC

Please note, the clicks PAMGuard's Click Classifiery classified as porpoise clicks appear as 0 in both ClassifiedAs and ManualAsign fields. 

The equivalent files in pickle extension are the mat structures saved in a faster-to-read format for python. 
