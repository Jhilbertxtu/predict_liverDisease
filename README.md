# predict_liverDisease
An end to end Machine learning project where the task is to classify whether a person has a liver disease or not .  
The dataset contains  contains 416 liver patient records and 167 non liver patient records collected from North East of Andhra Pradesh, India. The "Dataset" column is a class label used to divide groups into liver patient (liver disease) or not (no disease). This data set contains 441 male patient records and 142 female patient records.

Any patient whose age exceeded 89 is listed as being of age "90".

Columns:

Age of the patient
Gender of the patient
Total Bilirubin
Direct Bilirubin
Alkaline Phosphotase
Alamine Aminotransferase
Aspartate Aminotransferase
Total Protiens
Albumin
Albumin and Globulin Ratio
Dataset: field used to split the data into two sets (patient with liver disease, or no disease)

I used Logistic regression as the primary classifier, achieving the follow results:
accuracy : approx 73% in the validation set.
precision : approx 80%
recall: approx 92%
F1 Score : approx 85%
