import torch
from torchmetrics import MetricCollection, Metric


class MetricChallangeIABiodiv(Metric):
    
    def __init__(self):
        super().__init__() 
        # Initialisation des accumulateurs 
        self.species = self.add_state("sum_of_dif", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.diff = self.add_state("n_occ", default=torch.tensor(0.0), dist_reduce_fx="sum")
        
    def update(self, y_hat, y):
        y_temp=torch.clone(y)
        # repasser les valleur en comptage
        #y_temp=torch.pow(10, y_temp)-1
        # revient à faire max(1-y)
        y_temp[y_temp<1]=1

        y_hat_temp=torch.clone(y_hat)
        # repasser les valleur en comptage
        #y_hat_temp=torch.pow(10, y_hat_temp)-1
        # revient à faire max(1-ŷ)
        y_hat_temp[y_hat_temp<1]=1 
        
        # la somme de la valeur absolue de log10(ŷ)-log10(y)
        sum_of_dif=torch.sum(torch.abs(torch.log10(y_hat_temp)-torch.log10(y_temp)))        
        # le nombre d'occurence
        n_occ = y_temp.numel()
        # Mise à jour des accumulateurs
        self.sum_of_dif += sum_of_dif
        self.n_occ += n_occ
    
    def compute(self):
        # Calcul de la métrique finale
        return self.sum_of_dif / self.n_occ