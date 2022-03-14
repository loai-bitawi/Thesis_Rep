from config import default_settings
from Preprocessing import preprocessing
from Density_algorithm import density
from results import result
class main():
    def __init__(self):
        default=default_settings()
        self.main_path=default.main_path
        default.importer()
        self.preprocessor=preprocessing()
        
    def run_main(self,dataset,flag,filename,job_type='Multithreading',njobs=2,topj=10,model_name="bert-base-uncased",sample=False,sample_size=-1,stop_words=True):
        if dataset.lower()=='lisa':
            print('LISA dataset processing started...')
            self.preprocessor.lisa_preprocessing(sample=sample,sample_size=sample_size,stop_words=stop_words)
            print('Preprocessing Finished.')
            self.algorithm=density(flag=flag,dataset=dataset,filename=filename)
            self.algorithm.runner(job_type=job_type,njobs=njobs,topj=topj)
            print('Ranking Finished')
        elif dataset.lower()=='aila':
            self.preprocessor.aila_preprocessing(sample=sample,sample_size=sample_size,stop_words=stop_words)
            self.algorithm=density(flag=flag,dataset=dataset,filename=filename)
            self.algorithm.runner(job_type=job_type,njobs=njobs,topj=topj)
        elif dataset.lower()=='cranfield':
            self.preprocessor.cran_preprocessing(sample=sample,sample_size=sample_size,stop_words=stop_words)
            self.algorithm=density(flag=flag,dataset=dataset,filename=filename)
            self.algorithm.runner(job_type=job_type,njobs=njobs,topj=topj)            
        res=result(dataset=dataset,flag=flag)
        scores,aggrigated_scores=res.scores()
        return scores,aggrigated_scores
    
    
x=main()
scores,aggrigated_scores= x.run_main(dataset='lisa',flag='Euclidean',filename='docs.txt',stop_words=False) 
scores.to_excel(x.main_path+'/lisa/'+'scores.xlsx')
aggrigated_scores.to_excel(x.main_path+'/lisa/'+'aggrigated_scores.xlsx')