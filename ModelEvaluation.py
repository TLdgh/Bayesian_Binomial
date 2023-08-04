import pandas as pd
import numpy as np
import scipy.stats as stats
import math
import plotly.express as px

class Bayes():
    def __init__(self, data: dict, threshold: list, alpha, beta, alphalevel, plot=False, direction="less"):
        self.data=data
        self.threshold = threshold
        self.alpha = alpha
        self.beta = beta
        self.alphalevel=alphalevel
        
        self.res=self.CI_Bayes(direction)
        
        if plot:
            self.Plot_Bayes()
            
        
    def CI_Bayes(self, direction):
        res = []
        
        for item, j, a, b in zip(self.data, self.threshold, self.alpha, self.beta):
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
            res.append([round(param_1/(a+b+n),4), round(((a+x)*(b+n-x))/(pow(a+b+n,2)*(a+b+n+1)),6), CI])
        
        res=pd.DataFrame(res, columns=['Expected Success Probability', 'Variance of Success Probability', f'{(1-self.alphalevel)*100}%-level Credible Interval'])
        res.insert(0, 'Component', list(self.data.keys()))
        return(res)


    def Plot_Bayes(self, yrange=[0,1]):
        CI_upper=[x[1]-y  for x, y in zip(self.res[f'{(1-self.alphalevel)*100}%-level Credible Interval'], self.res['Expected Success Probability'])]
        CI_lower=[y-x[0]  for x, y in zip(self.res[f'{(1-self.alphalevel)*100}%-level Credible Interval'], self.res['Expected Success Probability'])]

        fig=px.bar(self.res, 
                x ='Component', y='Expected Success Probability', 
                text=[str("{:.2%}".format(x)) for x in self.res['Expected Success Probability']])
        
        fig.update_yaxes(range=yrange, dtick=0.05)
        fig.update_traces(textfont_size=12, textposition="outside",
                        error_y={"type": "data", 
                                "symmetric": False,
                                "array": CI_upper,
                                "arrayminus": CI_lower})
        fig.update_layout(width=1200, height=700,
                        title=dict(text=f"Expected Success Probability by Category with {(1-self.alphalevel)*100}% Confidence Level", x=0.5))
        fig.show()
