import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="ticks", color_codes=True)
sns.set_style("whitegrid")
sns.set_context("poster")


def show_data():
    """
    Plots features pairwise, of the following set:
    - average allelic fraction
    - hematocrit
    - platelet
    - white blood cell count
    - hemoglobin
    - age
    """

    # Put data into pandas dataframe
    aml_data = pd.read_csv('data.csv', index_col=0)

    # Convert everything to numeric values
    del aml_data['DrawID']
    d = {'Yes': 1, 'No': -1, 'caseflag': 'caseflag'}
    aml_data['caseflag'].replace(d, inplace=True)

    # Fill in missing values
    for column in aml_data.columns:
        aml_data[column].fillna(aml_data[column].mean(), inplace=True)

    # Add all gene columns togther, then delete them
    aml_data['Total'] = aml_data['Gene.1']
    for i in range(2, 46):
        col_header = 'Gene.'+str(i)
        aml_data['Total'] += aml_data[col_header]
    for i in range(1, 46):
        col_header = 'Gene.' + str(i)
        del aml_data[col_header]

    # Plot pairwise
    sns.set(style='whitegrid')
    cols = ['caseflag', 'Total', 'Age', 'WBC', 'PLATELET', 'HEMOGLBN', 'HEMATOCR']
    sns.pairplot(aml_data[cols],
                 hue='caseflag',  # different caseflags have different colors
                 markers=['.', r'$+$'],  # markers
                 palette=['#22d822', '#8A2BE2'],  # -, + colors in hex
                 plot_kws={"s": 250},  # marker size (100 default)
                 size=5.0)  # size of each subplot
    plt.show()
