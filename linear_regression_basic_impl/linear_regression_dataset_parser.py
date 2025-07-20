from dataset_downloader.download_kaggle_dataset import KaggleDataSet


class DataSetParser:
    def parse(self, dataset_slug, file_name_in_dataset, x_train_features, y_train_label):
        kaggle_dataset = KaggleDataSet()
        df = kaggle_dataset.download(dataset_slug, file_name_in_dataset)
        x_train_df = df[x_train_features]
        y_train_df = df[y_train_label]
        return x_train_df, y_train_df, df

