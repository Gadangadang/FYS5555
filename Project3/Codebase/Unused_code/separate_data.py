import os 


main_file = open("../data/mcdata.csv", "r")
#weights = open("../data/mcWeight_data.csv", "r")


def create_features_list(main_file):
    features = main_file.readline().split(",")[1:] # Skip the first line with features
    features = [feat.replace("\n", "") for feat in features]

    with open("../data/feature_names.csv", "w") as f:
        for feature in features[:-1]:
            f.write(feature)
            f.write(",")
        f.write(features[-1]+"\n")
    


def create_sample_set(main_file):
    # Ensure that the first line of the datafile is read, 
    # and that the features are kept in different files
    create_features_list(main_file)


    #Create mini sample sets
    start_val = 0
    amount = int(7e6)
    num_set = 0

    new_file = open("../data/dataset_{}.csv".format(num_set), "w")

    for line in main_file:
        
        for data in line[:-1]:
            new_file.write(data)
            new_file.write(",")
        new_file.write(line[-1])

        if start_val == amount:
            
            new_file.close()
            start_val = 0
            num_set += 1
            new_file = open("../data/dataset_{}.csv".format(num_set), "w")
            break
    
    new_file.close()
    #weights.close()
    main_file.close()


if __name__ == "__main__":

    file = open("test.txt", "w")
    file.write(str(1))
    file.close()
    
    #create_sample_set(main_file)