from config import default_settings
from Preprocessing import preprocessing
from Density_algorithm import density
from results import result
class main():
    def __init__(self):
        default=default_settings()
        default.importer()
        self.preprocessor=preprocessing()
        
    def run_main(self,dataset,flag,filename,job_type='Multithreading',njobs=2,topj=10,model_name="bert-base-uncased",sample=False,sample_size=-1):
        if dataset.lower()=='lisa':
            self.preprocessor.lisa_preprocessing(sample=sample,sample_size=sample_size)
            self.algorithm=density(flag=flag,dataset=dataset,filename=filename)
            self.algorithm.runner(job_type=job_type,njobs=njobs,topj=topj)
        elif dataset.lower()=='aila':
            self.preprocessor.aila_preprocessing(sample=sample,sample_size=sample_size)
            self.algorithm=density(flag=flag,dataset=dataset,filename=filename)
            self.algorithm.runner(job_type=job_type,njobs=njobs,topj=topj)
        elif dataset.lower()=='cranfield':
            self.preprocessor.cran_preprocessing(sample=sample,sample_size=sample_size)
            self.algorithm=density(flag=flag,dataset=dataset,filename=filename)
            self.algorithm.runner(job_type=job_type,njobs=njobs,topj=topj)            
        res=self.result(dataset=dataset,flag=flag)
        scores,aggrigated_scores=res.scores()
        return scores,aggrigated_scores
            
scores,aggrigated_scores= main().run_main(dataset='cranfield',flag='Euclidean',filename='docs.txt')
        
