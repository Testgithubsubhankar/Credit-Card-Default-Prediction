from creditcard.constant import COLUMN_LIMIT_BAL,COLUMN_AGE,COLUMN_BILL_AMT1,COLUMN_BILL_AMT2,COLUMN_BILL_AMT3,COLUMN_BILL_AMT4,COLUMN_BILL_AMT5,COLUMN_BILL_AMT6 ,COLUMN_PAY_AMT1,COLUMN_PAY_AMT2,COLUMN_PAY_AMT3,COLUMN_PAY_AMT4,COLUMN_PAY_AMT5,COLUMN_PAY_AMT6
from creditcard.exception import CreditcardException
from creditcard.logger import logging
from creditcard.entity.config_entity import DatTransformationConfig
from creditcard.entity.artifact_entity import DataIngestionArtifact,\
DataValidationArtifact,DataTransformationArtifact
import sys,os
import numpy as np
from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
import pandas as pd
from creditcard.constant import *
from creditcard.util.util import read_yaml_file,save_object,save_numpy_array_data,load_data





   # ID: float
   # LIMIT_BAL: float
    #SEX: category
    #EDUCATION: category
   # MARRIAGE: category
    #AGE: float
    #PAY_0: category
    #PAY_2: category
    #PAY_3: category
    #PAY_4: category
    #PAY_5: category
    #PAY_6: category
    #BILL_AMT1: float
    #BILL_AMT2: float
    #BILL_AMT3: float
    #BILL_AMT4: float
    #BILL_AMT5: float
    #BILL_AMT6: float
    #PAY_AMT1: float
    #PAY_AMT2: float
    #PAY_AMT3: float
    #PAY_AMT4: float
    #PAY_AMT5: float
    #PAY_AMT6: float
    #default.payment.next.month: category


class FeatureGenerator(BaseEstimator, TransformerMixin):

    def __init__(self,limit_bal_ix=1,
                 age_ix=5,
                 bill_amt1_ix=12,
                 bill_amt2_ix=13,
                 bill_amt3_ix=14,
                 bill_amt4_ix=15,
                 bill_amt5_ix=16,
                 bill_amt6_ix=17,
                 pay_amt1_ix=18,
                 pay_amt2_ix=19,
                 pay_amt3_ix=20,
                 pay_amt4_ix=21,
                 pay_amt5_ix=22,
                 pay_amt6_ix=23, columns=None):
        """
        FeatureGenerator Initialization
        limit_bal_ix: int index number of limit_bal columns
        age_ix: int index number of age columns
        bill_amt1_ix: int index number of  bill_amt1 columns
        bill_amt2_ix: int index number of bill_amt2 columns
        bill_amt3_ix: int index number of bill_amt3 columns
        bill_amt4_ix: int index number of bill_amt4 columns
        bill_amt5_ix: int index number of bill_amt5 columns
        bill_amt6_ix: int index number of bill_amt6 columns
        pay_amt1_ix: int index number of pay_amt1 columns
        pay_amt2_ix: int index number of pay_amt2 columns
        pay_amt3_ix: int index number of pay_amt3 columns
        pay_amt4_ix: int index number of pay_amt4 columns
        pay_amt5_ix: int index number of pay_amt5 columns
        pay_amt6_ix: int index number of pay_amt6 columns
        """
        try:
            self.columns = columns
            if self.columns is not None:
                limit_bal_ix = self.columns.index(COLUMN_LIMIT_BAL)
                age_ix = self.columns.index(COLUMN_AGE)
                bill_amt1_ix = self.columns.index(COLUMN_BILL_AMT1)
                bill_amt2_ix = self.columns.index(COLUMN_BILL_AMT2)
                bill_amt3_ix = self.columns.index(COLUMN_BILL_AMT3)
                bill_amt4_ix = self.columns.index(COLUMN_BILL_AMT4)
                bill_amt5_ix = self.columns.index(COLUMN_BILL_AMT5)
                bill_amt6_ix = self.columns.index(COLUMN_BILL_AMT6)
                pay_amt1_ix = self.columns.index(COLUMN_PAY_AMT1)
                pay_amt2_ix = self.columns.index(COLUMN_PAY_AMT2)
                pay_amt3_ix = self.columns.index(COLUMN_PAY_AMT3)
                pay_amt4_ix = self.columns.index(COLUMN_PAY_AMT4)
                pay_amt5_ix = self.columns.index(COLUMN_PAY_AMT5)
                pay_amt6_ix = self.columns.index(COLUMN_PAY_AMT6)

            
            self.limit_bal_ix = limit_bal_ix
            self.age_ix  = age_ix 
            self.bill_amt1_ix = bill_amt1_ix
            self.bill_amt2_ix = bill_amt2_ix
            self.bill_amt3_ix =bill_amt3_ix
            self.bill_amt4_ix = bill_amt4_ix
            self.bill_amt5_ix =bill_amt5_ix
            self.bill_amt6_ix = bill_amt6_ix
            self.pay_amt1_ix =pay_amt1_ix
            self.pay_amt2_ix =pay_amt2_ix
            self.pay_amt3_ix = pay_amt3_ix
            self.pay_amt4_ix = pay_amt4_ix
            self.pay_amt5_ix = pay_amt5_ix
            self.pay_amt6_ix = pay_amt6_ix



        except Exception as e:
            raise CreditcardException(e, sys) from e

   

    








class DataTransformation:

    def __init__(self, data_transformation_config:DatTransformationConfig,
                 data_ingestion_artifact: DataIngestionArtifact,
                 data_validation_artifact: DataValidationArtifact
                 ):
        try:
            
            logging.info(f"{'>>' * 30}Data Transformation log started.{'<<' * 30} ")
            self.data_transformation_config= data_transformation_config
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_artifact = data_validation_artifact

        except Exception as e:
            raise CreditcardException(e,sys) from e


    def get_data_transformer_object(self)->ColumnTransformer:
        try:
            schema_file_path = self.data_validation_artifact.schema_file_path

            dataset_schema = read_yaml_file(file_path=schema_file_path)

            numerical_columns = dataset_schema[NUMERICAL_COLUMN_KEY]
            categorical_columns = dataset_schema[CATEGORICAL_COLUMN_KEY]


            num_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy="median")),
                ('feature_generator', FeatureGenerator(
                    columns=numerical_columns
                )),
                ('scaler', StandardScaler())
            ]
            )

            cat_pipeline = Pipeline(steps=[
                 ('impute', SimpleImputer(strategy="most_frequent")),
                 ('one_hot_encoder', OneHotEncoder()),
                 ('scaler', StandardScaler(with_mean=False))
            ]
            )

            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")


            preprocessing = ColumnTransformer([
                ('num_pipeline', num_pipeline, numerical_columns),
                ('cat_pipeline', cat_pipeline, categorical_columns),
            ])
            return preprocessing

        except Exception as e:
            raise CreditcardException(e,sys) from e   


    def initiate_data_transformation(self)->DataTransformationArtifact:
        try:
            logging.info(f"Obtaining preprocessing object.")
            preprocessing_obj = self.get_data_transformer_object()


            logging.info(f"Obtaining training and test file path.")
            train_file_path = self.data_ingestion_artifact.train_file_path
            test_file_path = self.data_ingestion_artifact.test_file_path
            

            schema_file_path = self.data_validation_artifact.schema_file_path
            
            logging.info(f"Loading training and test data as pandas dataframe.")
            train_df = load_data(file_path=train_file_path, schema_file_path=schema_file_path)
            
            test_df = load_data(file_path=test_file_path, schema_file_path=schema_file_path)

            schema = read_yaml_file(file_path=schema_file_path)

            target_column_name = schema[TARGET_COLUMN_KEY]


            logging.info(f"Splitting input and target feature from training and testing dataframe.")
            input_feature_train_df = train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df = test_df[target_column_name]
            

            logging.info(f"Applying preprocessing object on training dataframe and testing dataframe")
            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)


            train_arr = np.c_[ input_feature_train_arr, np.array(target_feature_train_df)]

            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]
            
            transformed_train_dir = self.data_transformation_config.transformed_train_dir
            transformed_test_dir = self.data_transformation_config.transformed_test_dir

            train_file_name = os.path.basename(train_file_path).replace(".csv",".npz")
            test_file_name = os.path.basename(test_file_path).replace(".csv",".npz")

            transformed_train_file_path = os.path.join(transformed_train_dir, train_file_name)
            transformed_test_file_path = os.path.join(transformed_test_dir, test_file_name)

            logging.info(f"Saving transformed training and testing array.")
            
            save_numpy_array_data(file_path=transformed_train_file_path,array=train_arr)
            save_numpy_array_data(file_path=transformed_test_file_path,array=test_arr)

            preprocessing_obj_file_path = self.data_transformation_config.preprocessed_object_file_path

            logging.info(f"Saving preprocessing object.")
            save_object(file_path=preprocessing_obj_file_path,obj=preprocessing_obj)

            data_transformation_artifact = DataTransformationArtifact(is_transformed=True,
            message="Data transformation successfull.",
            transformed_train_file_path=transformed_train_file_path,
            transformed_test_file_path=transformed_test_file_path,
            preprocessed_object_file_path=preprocessing_obj_file_path

            )
            logging.info(f"Data transformationa artifact: {data_transformation_artifact}")
            return data_transformation_artifact
        except Exception as e:
            raise CreditcardException(e,sys) from e

    def __del__(self):
        logging.info(f"{'>>'*30}Data Transformation log completed.{'<<'*30} \n\n")
