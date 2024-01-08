
import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

data = pd.read_csv("aug_train.csv")

data["major_discipline"].replace(["Business Degree", "No Major"],
                             ["Business_Degree","No_Major"],inplace=True)

data["enrolled_university"].replace(["Full time course", "Part time course"],
                             ['Full_time_course','Part_time_course'],inplace=True)

data["education_level"].replace(["High School", "Primary School"],
                             ['High_School','Primary_School'],inplace=True)

data["company_type"].replace(["Pvt Ltd","Funded Startup","Public Sector","Early Stage Startup"],
                             ["Pvt_Ltd","Funded_Startup","Public_Sector","Early_Stage_Startup"],inplace=True)

data["company_size"].replace(["<10","10/49", "50-99", "100-500", "500-999", "1000-4999", "5000-9999", "10000+"],
                             ["Startup","Small","Small","Medium","Medium","Large","Large","Large"],inplace=True)

data["relevent_experience"].replace(["Has relevent experience", "No relevent experience"],
                             ['Yes','No'],inplace=True)

data["last_new_job"].replace([">4","never"],
                        ['5',"0"],inplace=True)