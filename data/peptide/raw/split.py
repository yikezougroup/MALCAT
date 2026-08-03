import pandas as pd
from sklearn.model_selection import train_test_split


def split_dataset(
    input_file: str = "data.csv",
    train_file: str = "data_train.csv",
    test_file: str = "data_test.csv",
    test_size: float = 0.2,
    random_seed: int = 42,
) -> None:
    """将CSV数据随机划分为训练集和测试集。"""

    # 读取数据
    data = pd.read_csv(input_file)

    if data.empty:
        raise ValueError(f"{input_file} 中没有数据。")

    # 8:2随机划分
    train_data, test_data = train_test_split(
        data,
        test_size=test_size,
        random_state=random_seed,
        shuffle=True,
    )

    # 保存结果，不写入额外的行索引
    train_data.to_csv(train_file, index=False)
    test_data.to_csv(test_file, index=False)

    print(f"原始数据量：{len(data)}")
    print(f"训练集数据量：{len(train_data)}，占比：{len(train_data) / len(data):.2%}")
    print(f"测试集数据量：{len(test_data)}，占比：{len(test_data) / len(data):.2%}")
    print(f"训练集已保存至：{train_file}")
    print(f"测试集已保存至：{test_file}")


if __name__ == "__main__":
    split_dataset()