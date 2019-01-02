import h5py
with h5py.File('fashon_parsing_data.mat', 'r') as file:
    print(list(file.keys()))
    # print(file)
    # ['#refs#', 'all_category_name', 'all_colors_name', 'fashion_dataset']

    for each in file.keys():
        try:
            print(file[each])
            print(file[each][0])
            print(file[each][0][0])

            for every in file[each][0]:
                st = every
                obj = file[st]
                str1 = ''.join(chr(i) for i in obj)
                print(str1)
        except Exception as e:
            print(e)

    print(file['fashion_dataset'])
    print(file['fashion_dataset'][0])
    print(file['fashion_dataset'][0][0])
    for every in file['fashion_dataset'][0]:
        st = every
        obj = file[st]
        for i in obj:
            print(i)
