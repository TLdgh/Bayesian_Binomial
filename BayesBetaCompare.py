import scipy.stats as stats
import numpy as np
import pandas as pd
import plotly.express as px


class CompareReview:
    def __init__(self, data: list, nExperiments=1000000, lev=0.95):
        self.data = data
        self.nExperiments = nExperiments
        self.lev = lev
        self.figs = []
        self.Probs = []
        ContingencyT = pd.DataFrame(np.zeros((len(data), len(data))))

        for i in range(len(data)):
            for j in range(len(data)):
                if i >= j:
                    ContingencyT.iloc[i, j] = "---"
                else:
                    ContingencyT.iloc[i, j] = self.simulateP(
                        data[i], data[j], nExperiments, lev
                    )

        ContingencyT.columns = ["Data" + str(i) for i in range(len(data))]
        ContingencyT.index = ["Data" + str(i) for i in range(len(data))]
        self.ContingencyT = ContingencyT

    def simulateP(self, data1: list, data2: list, nExperiments, lev):
        x1 = stats.binom.rvs(p=data1[0], n=data1[1], size=nExperiments)
        a1 = 1 + x1
        b1 = 1 + data1[1] - x1

        x2 = stats.binom.rvs(p=data2[0], n=data2[1], size=nExperiments)
        a2 = 1 + x2
        b2 = 1 + data2[1] - x2

        Probability = sum(
            [
                stats.beta.pdf(i, a2, b2) * stats.beta.cdf(i, a1, b1)
                for i in np.linspace(0, 1, 2000)
            ]
        )
        Probability = Probability / 2000
        quantil = np.quantile(Probability, lev)

        fig = px.histogram(dict(Probability=Probability), x="Probability", nbins=50)
        fig.update_layout(
            title=dict(text="Distribution of P(rate of X <= rate of Y)", x=0.5)
        )
        fig.add_vline(
            x=quantil,
            annotation_text=f"{lev*100}% quantile: {quantil*100}%",
            annotation_position="top left",
            annotation=dict(font_size=20),
        )

        self.Probs.append(Probability)
        self.figs.append(fig)
        return f"{round(quantil*100,2)}%"
