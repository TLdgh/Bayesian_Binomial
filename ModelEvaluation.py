import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import math
import plotly.express as px

class Bayes():
    def __init__(self, data: dict, threshold: list, alpha: list, beta: list, alphalevel, plot=False, direction="less"):
        self.data=data
        self.threshold = threshold
        self.alpha = alpha
        self.beta = beta
        self.alphalevel=alphalevel
        
        self.res=self.CI_Bayes(direction)
        
        if plot:
            self.Plot_Bayes()
            
        
    def CI_Bayes(self, direction):
        result = pd.DataFrame()
        
        for threshold_i in self.threshold:
            res=[]
            for item, j, a, b in zip(self.data, threshold_i, self.alpha, self.beta):
                values=self.data[item]
                n=len(values)

                if(direction=="less"):
                    x=len(list(filter(lambda x : x<=j, values))) 
                    param_1 = a+x
                    param_2 = b+n-x
                else:
                    x=len(list(filter(lambda x : x>=j, values))) 
                    param_1 = a+x
                    param_2 = b+n-x
                
                CI=stats.beta.ppf([self.alphalevel/2, 1-self.alphalevel/2], param_1, param_2).round(4)
                res.append([param_1, param_2, round(param_1/(param_1+param_2),4), round((param_1*param_2)/(pow(param_1+param_2,2)*(param_1+param_2+1)),6), CI, str(threshold_i)])
        
            res=pd.DataFrame(res, columns=['alpha', 'beta', 'Expected Success Probability', 'Variance of Success Probability', f'{(1-self.alphalevel)*100}%-level Credible Interval', 'Threshold'])
            res.insert(0, 'Component', list(self.data.keys()))
            result=pd.concat([result, res])
        
        return(result)


    def Plot_Bayes(self, yrange=[0,1, 0.05],width=1200, height=1200):
        CI_upper=[x[1]-y  for x, y in zip(self.res[f'{(1-self.alphalevel)*100}%-level Credible Interval'], self.res['Expected Success Probability'])]
        CI_lower=[y-x[0]  for x, y in zip(self.res[f'{(1-self.alphalevel)*100}%-level Credible Interval'], self.res['Expected Success Probability'])]

        fig=px.bar(self.res, 
                x ='Component', y='Expected Success Probability', color="Threshold", barmode = 'group',
                text=[str("{:.2%}".format(x)) for x in self.res['Expected Success Probability']])
        
        fig.update_yaxes(range=yrange[0:2], dtick=yrange[2])
        fig.update_traces(textfont_size=12, textposition="outside",
                        error_y={"type": "data", 
                                "symmetric": False,
                                "array": CI_upper,
                                "arrayminus": CI_lower})
        fig.update_layout(width=width, height=height,
                        title=dict(text=f"Expected Success Probability with {(1-self.alphalevel)*100}% Confidence Level", x=0.5))
        fig.show()
