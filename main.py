import warnings
warnings.filterwarnings("ignore")

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
        
    def run_main(self,dataset,flag,filename,job_type='Multithreading',njobs=2,topj=10,model_name="bert-base-uncased",sample=False,sample_size=-1,stop_words=True,preprocessing=True,hybrid=True,subfile='',tfidf_flag=True,old=False):
        if dataset.lower()=='lisa':
            print('\n LISA dataset processing started...\n')
            if preprocessing:
               self.preprocessor.lisa_preprocessing(sample=sample,sample_size=sample_size,stop_words=stop_words)
               print('Preprocessing Finished.')
            print('\nDensity Started\n')
            self.algorithm=density(flag=flag,dataset=dataset,filename=filename,stp_words=stop_words,subfile=subfile,tfidf_flag=tfidf_flag,hybrid=hybrid,old=old)
            self.algorithm.runner(job_type=job_type,njobs=njobs,topj=topj)
            print('Ranking Finished')
        elif dataset.lower()=='aila':
            print('\n AILA dataset processing started...\n')
            if preprocessing:
                self.preprocessor.aila_preprocessing(sample=sample,sample_size=sample_size,stop_words=stop_words)
                print('Preprocessing Finished.')
            self.algorithm=density(flag=flag,dataset=dataset,filename=filename,stop_words=stop_words)
            self.algorithm.runner(job_type=job_type,njobs=njobs,topj=topj)
            print('Ranking Finished')
        elif dataset.lower()=='cranfield':
            print('\n Cranfield dataset processing started...\n')
            if preprocessing:
                self.preprocessor.cran_preprocessing(sample=sample,sample_size=sample_size,stop_words=stop_words)
                print('Preprocessing Finished.')
            self.algorithm=density(flag=flag,dataset=dataset,filename=filename,stop_words=stop_words)
            self.algorithm.runner(job_type=job_type,njobs=njobs,topj=topj)            
            print('Ranking Finished')
        res=result(dataset=dataset,flag=flag)
        scores,aggrigated_scores=res.scores()
        return scores,aggrigated_scores
    
# x=main()
# scores,aggrigated_scores= x.run_main(dataset='lisa',njobs=32,job_type='Parallelism',flag='Euclidean',filename='docs.txt',stop_words=False,preprocessing=False,hybrid=False,subfile='without stop words removal') 
# scores.to_excel(x.main_path+'/lisa/'+'scores_Euclidean_no stp wrds.xlsx')
# aggrigated_scores.to_excel(x.main_path+'/lisa/'+'aggrigated_scores_Euclidean_no stp wrds rmv.xlsx')

    
# x=main()
# scores,aggrigated_scores= x.run_main(dataset='lisa',njobs=32,job_type='Parallelism',flag='Cosine',filename='docs.txt',stop_words=False,preprocessing=False,hybrid=False,subfile='without stop words removal') 
# scores.to_excel(x.main_path+'/lisa/'+'scores_cosine_no stp wrds.xlsx')
# aggrigated_scores.to_excel(x.main_path+'/lisa/'+'aggrigated_scores_cosine_no stp wrds rmv.xlsx')


    
# x=main()
# scores,aggrigated_scores= x.run_main(dataset='lisa',njobs=32,job_type='Parallelism',flag='Euclidean',filename='docs.txt',stop_words=True,preprocessing=False,hybrid=False,subfile='with stop words removal') 
# scores.to_excel(x.main_path+'/lisa/'+'scores_euclidean_with stp wrds rmvl.xlsx')
# aggrigated_scores.to_excel(x.main_path+'/lisa/'+'aggrigated_scores_euclidean_with stp wrds rmv.xlsx')

# x=main()
# scores,aggrigated_scores= x.run_main(dataset='lisa',njobs=32,job_type='Parallelism',flag='Cosine',filename='docs.txt',stop_words=True,preprocessing=False,hybrid=False,subfile='with stop words removal') 
# scores.to_excel(x.main_path+'/lisa/'+'scores_Cosine_with stp wrds rmvl.xlsx')
# aggrigated_scores.to_excel(x.main_path+'/lisa/'+'aggrigated_scores_Cosine_with stp wrds rmv.xlsx')



x=main()
scores,aggrigated_scores= x.run_main(dataset='lisa',flag='Euclidean',filename='sample27.txt',stop_words=True,preprocessing=False,tfidf_flag=False,hybrid=False,old=True) 
scores.to_excel(x.main_path+'/lisa/'+'scores_euclidean_with stp wrds rmvl_sample1.xlsx')
aggrigated_scores.to_excel(x.main_path+'/lisa/'+'aggrigated_scores_euclidean_with stp wrds rmv_sample1.xlsx')


#!!!! change density calculation (add tfidf)
x=main()
scores,aggrigated_scores= x.run_main(dataset='cranfield',flag='Euclidean',filename='docs.txt',stop_words=True,preprocessing=True,hybrid=False) 
scores.to_excel(x.main_path+'/cranfield/'+'scores_no hybrid.xlsx')
aggrigated_scores.to_excel(x.main_path+'/cranfield/'+'aggrigated_scores_no hybrid.xlsx')

x=main()
scores,aggrigated_scores= x.run_main(dataset='cranfield',flag='Euclidean',filename='docs.txt',stop_words=True,preprocessing=True,hybrid=True) 
scores.to_excel(x.main_path+'/cranfield/'+'scores_no hybrid.xlsx')
aggrigated_scores.to_excel(x.main_path+'/cranfield/'+'aggrigated_scores_no hybrid.xlsx')
