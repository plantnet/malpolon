import argparse
import pandas as pd


def find_trees(df):
     df_ano = pd.read_csv('PA_train_anonymized_glc24_private.csv', sep=';')
     df_nonano = df_nonano = pd.read_csv('Presences_Absences_train_glc24_nonAnonymized_new.csv', sep=';')
     df_ano = df_ano.sort_values(by='surveyID')
     df_nonano = df_nonano.sort_values(by='surveyID')
     df_all = df_ano.copy()
     df_all['species'] = df_nonano['species']
     jrc_trees = pd.read_csv('jrc_trees.csv')
     df_all_trees = df_all[df_all['species'].isin(jrc_trees['latin_name'])]
     tmp = df_all_trees[df_all_trees['surveyId'].isin(df['surveyId'])]
     df_trees = df[df['speciesId'].isin(df_all_trees['speciesId'])]
     return df_trees
     

def main(fp_df: str, sep):
    print(fp_df, sep)
    df = pd.read_csv(fp_df, sep=sep)
    df_trees = find_trees(df)
    print(df_trees.shape)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i",
                        nargs=1,
                        default='glc24_pa_train_CBN-med_matching-LUCAS-500m.csv',
                        help='Input path of the observation CSV.')
    parser.add_argument("--sep",
                        nargs=1,
                        default=[','],
                        type=str,
                        help='CSV separator')
    
    args = parser.parse_args()
    main(args.input, args.sep[0])
