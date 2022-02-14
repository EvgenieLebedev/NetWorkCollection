import os
extensions = {

    'data': ['json'],

    'sourse': ['source, source'],
    'mask_wda': ['wda'], 
    'mask_garbage': ['garbage']

}
main_path = 'D:\\wda-2021-03-28\\wda-2021-03-28'


class SortData(): #сортировка 
    #функция для создания папок из списка названий  
    def create_folders_from_list(folder_path, folder_names): 
        for folder in folder_names:
            if not os.path.exists(f'{folder_path}\\{folder}'):
                os.mkdir(f'{folder_path}\\{folder}') 
    #функция для получения путей подпапок
    def get_subfolder_paths(folder_path) -> list:
        subfolder_paths = [f.path for f in os.scandir(folder_path) if f.is_dir()]

        return subfolder_paths
    #функция для получения имен подпапок (не используется)
    def get_subfolder_names(folder_path) -> list:
        subfolder_paths = SortData.get_subfolder_paths(folder_path)
        subfolder_names = [f.split('\\')[-1] for f in subfolder_paths]

        return subfolder_names
    #функция пути всех файлов в папке
    def get_file_paths(folder_path) -> list:
        file_paths = [f.path for f in os.scandir(folder_path) if not f.is_dir()]

        return file_paths

    #функция получения имен файлов в папке
    def get_file_names(folder_path) -> list:
        file_paths = [f.path for f in os.scandir(folder_path) if not f.is_dir()]
        file_names = [f.split('\\')[-1] for f in file_paths]
        return file_names

    def get_file_names_numbers(folder_path) -> list:
        file_paths = [f.path for f in os.scandir(folder_path) if not f.is_dir()]
        file_names = [f.split('\\')[-1] for f in file_paths]
        file_names = [f.split('.')[0] for f in file_names]
        return file_names

    #функция сортировки
    def sort_files(folder_path):
        file_paths = SortData.get_file_paths(folder_path) # Получаем пути файлов
        ext_list = list(extensions.items()) #cписок метода словаря
        for file_path in file_paths:
            #extension = file_path.split('.')[-2] #вытаскиваем второе расширение (sourse, wda, garbage)
            extension = file_path.split('.')[-1] #вытаскиваем второе расширение (tiff, json)
            #print(extension)
            file_name = file_path.split('\\')[-1]

            for dict_key_int in range(len(ext_list)):
                 #Для каждого ключа в словаре мы проверяем, есть ли расширение файла в списке расширений.
                if extension in ext_list[dict_key_int][1]:
                    print(f'Moving {file_name} in {ext_list[dict_key_int][0]} folder\n')
                    os.rename(file_path, f'{main_path}\\{ext_list[dict_key_int][0]}\\{file_name}')

    #Удаляем пустые папки
    def remove_empty_folders(folder_path):
        subfolder_paths = SortData.get_subfolder_paths(folder_path)

        for p in subfolder_paths:
            if not os.listdir(p):
                print('Deleting empty folder:', p.split('\\')[-1], '\n')
                os.rmdir(p)

    #if __name__ == "__main__":
        #print("Начало работы программы")   
        #SortData.create_folders_from_list(main_path, extensions)
        #SortData.sort_files(main_path)
        #SortData.remove_empty_folders(main_path)
