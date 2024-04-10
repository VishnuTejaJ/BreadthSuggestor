import pandas as pd

with open("final_grades.csv", "r") as f:
    df = pd.read_csv(f)
    course = df.course.unique()
    df['sum'] = df[['EX','A','B','C','D','P','F']].sum(axis=1)
    df['EX%'] = df['EX'] / df['sum'] *100
    df['A%'] = df['A'] / df['sum'] *100
    df['B%'] = df['B'] / df['sum'] *100
    df['C%'] = df['C'] / df['sum'] *100
    df['D%'] = df['D'] / df['sum'] *100
    df['P%'] = df['P'] / df['sum'] *100
    df['F%'] = df['F'] / df['sum'] *100
    df['average'] = (10*df['EX']+9*df['A']+8*df['B']+7*df['C']+6*df['D']+5*df['P']+4*df['F'])/df['sum']
    df.drop(['sum'], axis=1, inplace=True)
    df.to_csv("final_grades.csv", index=False)
    
def get_priority(desired_department) :
    with open("final_grades.csv", "r") as f:
        df = pd.read_csv(f)
        df = df[df['course'].str[:2]==desired_department]
        df = df.sort_values(by='average', ascending=False)
        return df


# data_hs = get_priority("HS")[['course', 'average']]
# print(data_hs)
