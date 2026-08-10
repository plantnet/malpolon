import os
import argparse
import pandas as pd


def merge_2022_with_2018_and_before(df_2022, df_2006_2018):
    cols2022_to_keep = ['point_id', 'year', 'gps_lat', 'gps_long', 'gps_altitude', 'nuts0', 'file_path_gisco_north', 'file_path_gisco_south', 'file_path_gisco_east', 'file_path_gisco_west', 'file_path_gisco_point', 'file_path_ftp_cover']
    df_2022 = df_2022[cols2022_to_keep]
    df_2022 = df_2022.dropna(subset=['point_id', 'gps_lat', 'gps_long'])
    df_2022.rename(columns={'file_path_ftp_cover': 'file_path_gisco_cover'}, inplace=True)  # align 2022's cols with those of 2018's
    
    df_2006_2022 = pd.concat([df_2006_2018, df_2022])

    return df_2006_2022
    
def main(fp_2018, fp_2022, fp_out):
    df_2006_2018 = pd.read_csv(fp_2018)
    df_2022 = pd.read_csv(fp_2022)
    df_2006_2022 = merge_2022_with_2018_and_before(df_2006_2018, df_2022)

    df_2006_2022.to_csv(f'{fp_out}')
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_2022",
                        type=str,
                        help="Input LUCAS 2022 dataset.",)
    parser.add_argument("--input_2018",
                        type=str,
                        help="Input LUCAS 2006-2018 dataset.",)
    parser.add_argument("-o", "--output",
                        type=str,
                        default="LUCAS_2006-2022.csv",
                        help="Output file path.",)
    args = parser.parse_args()                        
    main(args.input_2022, args.input_2018, args.output)
