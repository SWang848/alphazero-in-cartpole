import matplotlib.pyplot as plt
import numpy as np


def visit_counts_distribution():
    # Generate sample data for 4 lists, each with 15 data points
    
    # list1 = [5,5,4,4,5,5,4,5,5,5,4,4,4,5,4]
    # list2 = [3,3,3,2,3,3,4,4,36,3,4,2,2,3,4]
    # list3 = [1,2,2,0,1,0,3,3,72,1,3,1,1,0,2]
    # list4 = [0,0,0,0,0,0,1,2,92,0,1,0,0,0,0]
    
    list1 = [5,6,4,4,5,5,5,5,4,5,4,4,4,5,4]
    list2 = [4,5,6,4,4,12,7,8,4,3,4,3,3,2,3]
    list3 = [6,1,6,6,6,15,10,4,1,6,7,5,6,1,6]
    list4 = [7,1,9,5,7,11,10,3,1,9,5,5,5,0,11]
    # Number of bars (actions)
    num_bars = len(list1)

    # X locations for the groups (actions)
    indices = np.arange(num_bars)

    # Bar width
    bar_width = 0.2

    # Creating the bar chart
    plt.figure(figsize=(12, 6))
    plt.bar(indices, list1, width=bar_width, label='model_0')
    plt.bar(indices + bar_width, list2, width=bar_width, label='model_10')
    plt.bar(indices + 2 * bar_width, list3, width=bar_width, label='model_20')
    plt.bar(indices + 3 * bar_width, list4, width=bar_width, label='model_30')

    # Labels and title
    plt.xlabel('Actions')
    plt.ylabel('Number of Visits')
    plt.title('Node Visit Distribution in the First Step of Search Trees')
    plt.xticks(indices + 1.5 * bar_width, ['a_12','a_19','a_29','a_30','a_31','a_40','a_41','a_49','a_51','a_52','a_53','a_60','a_62','a_63','a_70'], rotation=45)
    plt.legend()

    # Save the plot
    plt.savefig('distribution_2.png')

    # Close the plot to free memory
    plt.close()

if __name__ == '__main__':
    visit_counts_distribution()