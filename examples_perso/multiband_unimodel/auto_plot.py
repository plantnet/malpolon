import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

'''
Input : 
-> path_output_run : chemin absolu du dossier contenant le "metric.csv" en str, ce dossier est normalement situé dans le dossier outpout créé par malpolon
-> graph : si True tracera les graphes

Output : 
-> A partir du fichier "metric.csv", cette définition va :
    - tracer automatiquement des graphiques au format .png 
    - réaliser un csv "metric_lite.csv" plus simple d'utilisation que "metric.csv"
'''


def Autoplot(path_output_run, graph):
    path=Path(path_output_run)

    # exportation des données
    df = pd.read_csv(str(path) + '/metrics.csv', sep=',')

    # nombre d'époque complète
    n_epoch = int(df.shape[0]/4)

    # valeurs des index à récupérers pour la val
    list_val_index = []
    for i1 in range(2, 1+4*n_epoch, 4):
        list_val_index.append(i1)
    
    # valeurs des index à récupérers pour le train
    list_train_index = [x + 1 for x in list_val_index]
    
    # valeurs des index à récupérers pour le lr
    list_lr_index = [x - 2 for x in list_val_index]

    
    del df["step"]

        
    # mise en place d'un df des valeurs de val et de train par époque
    
    df_val = df.iloc[:,1:int((len(df.columns))/2+1)][df.index.isin(list_val_index)]
    df_train = df.iloc[:,int((len(df.columns))/2+0):][df.index.isin(list_train_index)]
    df_lr = pd.DataFrame({'lr': df[df.columns[0]][df.index.isin(list_lr_index)].tolist(),
                          'epoch': df_train['epoch'].tolist()})
        
    # mise en place du df propre final 
    df_final = pd.merge(df_lr, df_val, on='epoch').merge(df_train, on='epoch')
    temp=df_final.pop('epoch')
    df_final.insert(0, 'epoch', temp)

   
    df_final.train_loss
    # sauvegarde
    df_final.to_csv(str(path) + '/metrics_lite.csv', index=False)

    if graph == True:
        # création et sauvegarde des figures
        fig=plt.figure(figsize=[10,5], dpi=600)
        plt.plot(range(df_final.shape[0]), df_final.lr, label='lr')
        plt.title(path.name + ' - Learning rate')
        plt.ylabel("lr")
        plt.xlabel("epoch")
        plt.xticks(range(0,df_final.shape[0],2))
        plt.legend()
        fig.savefig(str(path) + '/learning_rate.png', bbox_inches='tight')
        
        fig=plt.figure(figsize=[10,5], dpi=600)
        plt.plot(range(df_final.shape[0]), df_final.train_loss, label='training loss')
        plt.plot(range(df_final.shape[0]), df_final.val_loss, label='validation loss')
        plt.title(path.name + ' - Loss values')
        plt.ylabel("loss")
        plt.xlabel("epoch")
        plt.xticks(range(0,df_final.shape[0],2))
        plt.legend()
        fig.savefig(str(path) + '/loss.png', bbox_inches='tight')
    
        if 'train_accuracy' in df.columns :
            fig=plt.figure(figsize=[10,5], dpi=600)
            plt.plot(range(df_final.shape[0]), df_final.train_accuracy, label='train', c='steelblue')
            plt.plot(range(df_final.shape[0]), df_final.val_accuracy, label='val', c='orange')
            plt.plot(range(df_final.shape[0]), df_final.train_accuracy_macro,"r--", label='train macro', c='steelblue')
            plt.plot(range(df_final.shape[0]), df_final.val_accuracy_macro,"r--" , label='val macro', c='orange')
            plt.title(path.name + ' - Accuracy top-k = 1')
            plt.ylabel("accuracy")
            plt.xlabel("epoch")
            plt.xticks(range(0,df_final.shape[0],2))
            plt.legend(bbox_to_anchor=[1.05,1],loc=2,borderaxespad=0)
            fig.savefig(str(path) + '/acc_top_1.png', bbox_inches='tight')

        if 'train_top_5_accuracy' in df.columns :
            fig=plt.figure(figsize=[10,5], dpi=600)
            plt.plot(range(df_final.shape[0]), df_final.train_top_5_accuracy, label='train', c='steelblue')
            plt.plot(range(df_final.shape[0]), df_final.val_top_5_accuracy, label='val', c='orange')
            plt.plot(range(df_final.shape[0]), df_final.train_top_5_accuracy_macro,"r--", label='train macro', c='steelblue')
            plt.plot(range(df_final.shape[0]), df_final.val_top_5_accuracy_macro,"r--" , label='val macro', c='orange')
            plt.title(path.name + ' - Accuracy top-k = 5')
            plt.ylabel("accuracy")
            plt.xlabel("epoch")
            plt.xticks(range(0,df_final.shape[0],2))
            plt.legend(bbox_to_anchor=[1.05,1],loc=2,borderaxespad=0)    
            fig.savefig(str(path) + '/acc_top_5.png', bbox_inches='tight')
   
        if 'train_top_10_accuracy' in df.columns :
            fig=plt.figure(figsize=[10,5], dpi=600)
            plt.plot(range(df_final.shape[0]), df_final.train_top_10_accuracy, label='train', c='steelblue')
            plt.plot(range(df_final.shape[0]), df_final.val_top_10_accuracy, label='val', c='orange')
            plt.plot(range(df_final.shape[0]), df_final.train_top_10_accuracy_macro,"r--", label='train macro', c='steelblue')
            plt.plot(range(df_final.shape[0]), df_final.val_top_10_accuracy_macro,"r--" , label='val macro', c='orange')
            plt.title(path.name + ' - Accuracy top-k = 10')
            plt.ylabel("accuracy")
            plt.xlabel("epoch")
            plt.xticks(range(0,df_final.shape[0],2))
            plt.legend(bbox_to_anchor=[1.05,1],loc=2,borderaxespad=0)   
            fig.savefig(str(path) + '/acc_top_10.png', bbox_inches='tight')
    
        if 'train_top_20_accuracy' in df.columns :
            fig=plt.figure(figsize=[10,5], dpi=600)
            plt.plot(range(df_final.shape[0]), df_final.train_top_20_accuracy, label='train', c='steelblue')
            plt.plot(range(df_final.shape[0]), df_final.val_top_20_accuracy, label='val', c='orange')
            plt.plot(range(df_final.shape[0]), df_final.train_top_20_accuracy_macro,"r--", label='train macro', c='steelblue')
            plt.plot(range(df_final.shape[0]), df_final.val_top_20_accuracy_macro,"r--" , label='val macro', c='orange')
            plt.title(path.name + ' - Accuracy top-k = 20')
            plt.ylabel("accuracy")
            plt.xlabel("epoch")
            plt.xticks(range(0,df_final.shape[0],2))
            plt.legend(bbox_to_anchor=[1.05,1],loc=2,borderaxespad=0)   
            fig.savefig(str(path) + '/acc_top_20.png', bbox_inches='tight')
    
        if 'train_top_30_accuracy' in df.columns :
            fig=plt.figure(figsize=[10,5], dpi=600)
            plt.plot(range(df_final.shape[0]), df_final.train_top_30_accuracy, label='train', c='steelblue')
            plt.plot(range(df_final.shape[0]), df_final.val_top_30_accuracy, label='val', c='orange')
            plt.plot(range(df_final.shape[0]), df_final.train_top_30_accuracy_macro,"r--", label='train macro', c='steelblue')
            plt.plot(range(df_final.shape[0]), df_final.val_top_30_accuracy_macro,"r--" , label='val macro', c='orange')
            plt.title(path.name + ' - Accuracy top-k = 30')
            plt.ylabel("accuracy")
            plt.xlabel("epoch")
            plt.xticks(range(0,df_final.shape[0],2))
            plt.legend(bbox_to_anchor=[1.05,1],loc=2,borderaxespad=0)   
            fig.savefig(str(path) + '/acc_top_30.png', bbox_inches='tight')
    
        if 'train_top_5_accuracy' in df.columns :
            fig=plt.figure(figsize=[10,5], dpi=600)
            plt.plot(range(df_final.shape[0]), df_final.train_accuracy, label='train top-k=1', c='red')
            plt.plot(range(df_final.shape[0]), df_final.val_accuracy, "r--", label='val top-k=1', c='red')
            plt.plot(range(df_final.shape[0]), df_final.train_top_5_accuracy, label='train top-k=5',c='sienna')
            plt.plot(range(df_final.shape[0]), df_final.val_top_5_accuracy, "r--" , label='val top-k=5',c='sienna')
            if 'train_top_10_accuracy' in df.columns :
                plt.plot(range(df_final.shape[0]), df_final.train_top_10_accuracy, label='train top-k=10', c='orange')
                plt.plot(range(df_final.shape[0]), df_final.val_top_10_accuracy, "r--", label='val top-k=10', c='orange')
            if 'train_top_20_accuracy' in df.columns :
                plt.plot(range(df_final.shape[0]), df_final.train_top_20_accuracy, label='train top-k=20', c='greenyellow')
                plt.plot(range(df_final.shape[0]), df_final.val_top_20_accuracy, "r--", label='val top-k=20', c='greenyellow')
            if 'train_top_30_accuracy' in df.columns :
                plt.plot(range(df_final.shape[0]), df_final.train_top_30_accuracy, label='train top-k=30', c='forestgreen')
                plt.plot(range(df_final.shape[0]), df_final.val_top_30_accuracy, "r--", label='val top-k=30', c='forestgreen')
            plt.title(path.name + ' - Accuracy values')
            plt.ylabel("accuracy")
            plt.xlabel("epoch")
            plt.xticks(range(0,df_final.shape[0],2))
            plt.legend(bbox_to_anchor=[1.05,1],loc=2,borderaxespad=0)   
            fig.savefig(str(path) + '/acc_blian.png', bbox_inches='tight')
    
        if 'train_top_5_accuracy' in df.columns :
            fig=plt.figure(figsize=[10,5], dpi=600)
            plt.plot(range(df_final.shape[0]), df_final.train_accuracy_macro, label='train top-k=1', c='red')
            plt.plot(range(df_final.shape[0]), df_final.val_accuracy_macro, "r--", label='val top-k=1', c='red')
            plt.plot(range(df_final.shape[0]), df_final.train_top_5_accuracy_macro, label='train top-k=5',c='sienna')
            plt.plot(range(df_final.shape[0]), df_final.val_top_5_accuracy_macro, "r--" , label='val top-k=5',c='sienna')
            if 'train_top_10_accuracy' in df.columns :
                plt.plot(range(df_final.shape[0]), df_final.train_top_10_accuracy_macro, label='train top-k=10', c='orange')
                plt.plot(range(df_final.shape[0]), df_final.val_top_10_accuracy_macro, "r--", label='val top-k=10', c='orange')
            if 'train_top_20_accuracy' in df.columns :
                plt.plot(range(df_final.shape[0]), df_final.train_top_20_accuracy_macro, label='train top-k=20', c='greenyellow')
                plt.plot(range(df_final.shape[0]), df_final.val_top_20_accuracy_macro, "r--", label='val top-k=20', c='greenyellow')
            if 'train_top_30_accuracy' in df.columns :
                plt.plot(range(df_final.shape[0]), df_final.train_top_30_accuracy_macro, label='train top-k=30', c='forestgreen')
                plt.plot(range(df_final.shape[0]), df_final.val_top_30_accuracy_macro, "r--", label='val top-k=30', c='forestgreen')
            plt.title(path.name + ' - Macro average accuracy values')
            plt.ylabel("macro average accuracy")
            plt.xlabel("epoch")
            plt.xticks(range(0,df_final.shape[0],2))
            plt.legend(bbox_to_anchor=[1.05,1],loc=2,borderaxespad=0)   
            fig.savefig(str(path) + '/acc_macro_blian.png', bbox_inches='tight')

        if 'train_metric_ia_biodiv' in df.columns :
            fig=plt.figure(figsize=[10,5], dpi=600)
            plt.plot(range(df_final.shape[0]), df_final.train_metric_ia_biodiv, label='train', c='steelblue')
            plt.plot(range(df_final.shape[0]), df_final.val_metric_ia_biodiv, label='val', c='orange')
            plt.title(path.name + ' - metric IA Biodiv')
            plt.ylabel("metric_ia_biodiv")
            plt.xlabel("epoch")
            plt.xticks(range(0,df_final.shape[0],2))
            plt.legend(bbox_to_anchor=[1.05,1],loc=2,borderaxespad=0)
            fig.savefig(str(path) + '/_metric_ia_biodiv.png', bbox_inches='tight')