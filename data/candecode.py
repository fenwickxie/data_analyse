import os
import platform
import can
import cantools
from typing import List, Dict, Any, Optional, Tuple, TypeAlias, Union
from cantools.database import Database
from tqdm import tqdm
from multiprocessing import Pool

StringPathLike: TypeAlias = Union[str, os.PathLike]

if platform.system() == "Windows":
    ENCODING = "gbk"
else:
    ENCODING = "utf-8"
class CanDecoder:
    def __init__(
        self, dbc_url: StringPathLike, can_url: StringPathLike
    ):  # 构造函数，初始化对象
        self.dbc_url = dbc_url  # 将传入的dbc_url参数赋值给对象的dbc_url属性
        self.can_url = can_url  # 将传入的can_url参数赋值给对象的can_url属性
        self.dbcs = self.__load_dbc_multi(
            dbc_url
        )  # 调用私有方法__load_dbc_multi加载dbc文件，并将结果赋值给对象的dbcs属性
        self.blf_urls, self.asc_urls = self.__load_can_multi(
            can_url
        )  # 调用私有方法__load_can_multi加载can文件，并将结果分别赋值给对象的blf_urls和asc_urls属性

    def __load_dbc_single(self, dbc_url: StringPathLike) -> Tuple[StringPathLike, cantools.database.Database]:
        """
        Load a DBC file and return the database object.
        """
        # 打开指定路径的DBC文件，以只读模式("r")和指定的编码格式(ENCODING)读取文件内容
        with open(dbc_url, "r", encoding=ENCODING) as f:
            # 使用cantools库的db模块加载DBC文件内容，指定文件格式为"dbc"，并设置严格模式为False
            dbc_content = cantools.db.load(f, database_format="dbc", strict=False)
        # 返回DBC文件的路径和加载的数据库对象
        return dbc_url, dbc_content

    def __load_dbc_multi(
        self,
        dbc_url: Union[StringPathLike, List[StringPathLike]],
    ) -> List[Tuple[str, Database]]:

        # 初始化一个空列表用于存储加载的数据库
        dbcs = []
        # 检查dbc_url的类型，如果是字符串路径
        if isinstance(dbc_url, StringPathLike):
            # 检查该路径是否是一个目录
            if os.path.isdir(dbc_url):
                # 获取目录下所有以.dbc结尾的文件路径
                dbc_urls = [
                    os.path.join(dbc_url, file)
                    for file in os.listdir(dbc_url)
                    if file.endswith(".dbc")
                ]
                # 使用map函数并行加载这些.dbc文件
                dbcs.extend(map(self.__load_dbc_single, dbc_urls))

            # 如果是一个文件
            elif os.path.isfile(dbc_url):
                # 将该文件路径添加到列表中
                dbc_urls = [dbc_url]
                # 注释掉的代码：原本是直接调用__load_dbc_single函数加载单个文件
                # dbcs.append(__load_dbc_single(dbc_url))
            else:
                # 如果既不是目录也不是文件，抛出异常
                raise ValueError(f"Invalid DBC file path: {dbc_url}")
        # 如果dbc_url是列表
        elif isinstance(dbc_url, list):
            # 过滤出所有以.dbc结尾的文件路径
            dbc_urls = [url for url in dbc_url if url.endswith(".dbc")]
            # 注释掉的代码：原本是直接调用__load_dbc_single函数加载列表中的文件
            # dbcs.extend(map(__load_dbc_single, dbc_url))
        else:
            # 如果dbc_url既不是字符串也不是列表，抛出异常
            raise ValueError(f"Invalid DBC file path: {dbc_url}")
        # 导入ThreadPoolExecutor用于并行处理
        from concurrent.futures import ThreadPoolExecutor

        # 使用ThreadPoolExecutor并行加载所有.dbc文件
        with ThreadPoolExecutor() as executor:
            dbcs = list(executor.map(self.__load_dbc_single, dbc_urls))
        # 返回加载的数据库列表
        return dbcs

    def __load_can_multi(
        self,
        can_url: Union[StringPathLike, List[StringPathLike]],
    ) -> Tuple[List[StringPathLike], List[StringPathLike]]:
        """
        加载多个 CAN 文件路径，并根据文件类型（.blf 或 .asc）分类。

        Args:
            can_url (Union[StringPathLike, List[StringPathLike]]): CAN 文件路径或目录路径，或包含多个路径的列表。

        Returns:
            Tuple[List[StringPathLike], List[StringPathLike]]: 包含 .blf 文件路径列表和 .asc 文件路径列表的元组。
        """
        blf_urls = []  # 存储所有 .blf 文件路径的列表
        asc_urls = []  # 存储所有 .asc 文件路径的列表

        def __process_path(path: StringPathLike):
            """处理单个路径，分类为 .blf 或 .asc 文件"""
            if os.path.isdir(path):
                # 如果路径是目录，列出目录中的所有文件
                files = os.listdir(path)
                # 使用列表推导式和过滤器一次性处理文件
                files = os.listdir(path)
                blf_urls.extend(
                    os.path.join(path, file) for file in files if file.endswith(".blf")
                )
                asc_urls.extend(
                    os.path.join(path, file) for file in files if file.endswith(".asc")
                )
            elif os.path.isfile(path):
                # 使用字典映射减少 if-else 判断
                extension_map = {".blf": blf_urls, ".asc": asc_urls}
                ext = os.path.splitext(path)[1].lower()
                if ext in extension_map:
                    extension_map[ext].append(path)
            return blf_urls, asc_urls

        # 单个路径
        if isinstance(can_url, (str, os.PathLike)):
            __process_path(can_url)
        # 列表使用多线程并行处理
        elif isinstance(can_url, list):
            from concurrent.futures import ThreadPoolExecutor

            with ThreadPoolExecutor() as executor:
                executor.map(__process_path, can_url)
        else:
            raise ValueError(
                "can_url must be a string, PathLike, or a list of such objects."
            )

        return blf_urls, asc_urls

    def __decode_can(
        self,
        dbc_data,
        can_data,
        signal_names: Optional[List[StringPathLike]] = None,
        signal_corr: Optional[Dict[str, str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Decode CAN data using the provided DBC data.
        """
        from asammdf import Signal  # 从 asammdf 库导入 Signal 类

        decoded = {}  # 初始化一个空字典用于存储解码后的信号数据
        signal_names_set = (
            set(signal_names) if signal_names else None
        )  # 使用集合加速查找
        try:
            for __msg in can_data:
                try:
                    __dec = dbc_data.decode_message(
                        __msg.arbitration_id,
                        __msg.data,
                    )
                    if not __dec:
                        continue

                    for __k, __v in __dec.items():
                        # 如果 signal_names 是 None 或 __k 在 signal_names 中
                        if signal_names_set is None or __k in signal_names_set:
                            if __k not in decoded:
                                decoded[__k] = []
                            # 使用 getattr(__v, "value", __v)尝试获取 __v 的值，如果失败则使用 __v 本身
                            value = getattr(__v, "value", __v)
                            decoded[__k].append((__msg.timestamp, value))
                except Exception:
                    continue
        except TypeError:
            raise TypeError("can_data must be an iterable object")

        # 构建 Signal 对象
        sigs = []
        for __k, __v in decoded.items():
            __time, __data = zip(*__v)  # 使用 zip 解压时间戳和数据
            if signal_corr and __k in signal_corr:
                __k = signal_corr[__k]
            sigs.append(Signal(list(__data), list(__time), name=__k, encoding="utf-8"))

        return sigs

    def __save_to(
        self,
        dbc_file_url,
        can_file_url,
        signals,
        step: float = 0.002,
        save_dir: StringPathLike = r"./can_decoded",
        save_formats: Tuple[str, ...] = (".csv", ".parquet", ".mat"),
    ):
        """
        Save decoded CAN data to specified formats.

        Args:
            dbc_file_url (str): Path to the DBC file.
            can_file_url (str): Path to the CAN file.
            signals (list): Decoded signals.
            step (float): Raster step size.
            save_dir (str): Directory to save the output files.
            save_formats (tuple): File formats to save (e.g., .csv, .parquet, .mat).
        """
        # 检查保存目录是否存在，如果不存在则创建
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # 导入asammdf库中的MDF类和scipy库中的io模块
        from asammdf import MDF
        import scipy.io as sio

        # 创建一个MDF对象
        mdf = MDF()
        # 将解码后的信号添加到MDF对象中
        mdf.append(signals)
        # 将MDF对象转换为DataFrame，指定栅格步长
        df = mdf.to_dataframe(raster=step)

        # 生成基础文件名，由DBC文件名和CAN文件名组合而成
        base_filename = (
            os.path.splitext(os.path.basename(dbc_file_url))[0]
            + "_"
            + os.path.splitext(os.path.basename(can_file_url))[0]
        )

        # 定义文件格式与保存方法的映射
        save_methods = {
            ".mat": lambda file_url: sio.savemat(file_url, df.to_dict(orient="list")),
            ".csv": lambda file_url: df.to_csv(file_url),
            ".parquet": lambda file_url: df.to_parquet(file_url),
        }

        # 遍历保存格式并调用对应的保存方法
        for save_format in save_formats:
            # 生成完整的文件路径
            __file_url = os.path.join(save_dir, f"{base_filename}{save_format}")
            # 获取对应的保存方法
            save_method = save_methods.get(save_format)
            if save_method:
                # 调用保存方法
                save_method(__file_url)
            else:
                # 如果不支持的文件格式，抛出异常
                raise ValueError(f"Unsupported save format: {save_format}")

    def read_single_can(
        self,
        dbc_url: str,
        dbc_data: Database,
        log_file_path: str,
        file_type: str,
        signal_names: Optional[List[str]] = None,
        signal_corr: Optional[Dict[str, str]] = None,
        step: float = 0.002,
        save_dir: str = r"./can_decoded",
        save_formats: Tuple[str, ...] = (".csv", ".parquet", ".mat"),
    ) -> List[Dict[str, Any]] | None:
        """
        Process a single CAN file (BLF or ASC) and save the decoded data.

        Args:
            dbc_url (str): Path to the DBC file.
            dbc_data (Database): DBC database object.
            log_file_path (str): Path to the CAN log file.
            file_type (str): Type of the CAN file ("blf" or "asc").
            signal_names (Optional[List[str]]): List of signal names to decode.
            signal_corr (Optional[Dict[str, str]]): Signal name corrections.
            step (float): Raster step size.
            save_dir (str): Directory to save the output files.
            save_formats (Tuple[str, ...]): File formats to save (e.g., ".csv", ".parquet").
        """
        try:
            # 根据文件类型加载日志数据
            if file_type == "blf":
                log_data = can.BLFReader(log_file_path)
            elif file_type == "asc":
                log_data = can.ASCReader(log_file_path)
            else:
                raise ValueError(f"Unsupported file type: {file_type}")

            # 解码信号
            signals = self.__decode_can(dbc_data, log_data, signal_names, signal_corr)

            # 保存解码结果
            self.__save_to(
                dbc_url, log_file_path, signals, step, save_dir, save_formats
            )
            return signals
        except Exception as e:
            print(f"Error processing file {log_file_path}: {e}")

    def read_can_files(
        self,
        signal_names: Optional[List[str]] = None,
        signal_corr: Optional[Dict[str, str]] = None,
        step: float = 0.002,
        save_dir: str = r"./can_decoded",
        save_formats: Tuple[str, ...] = (".csv", ".parquet", ".mat"),
    ) -> None:
        """
        Read CAN files and decode them using the provided DBC data (single-threaded).
        """

        # 确保保存目录存在
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # 遍历每个 DBC 文件
        for __dbc_url, __dbc_data in self.dbcs:
            # 处理 BLF 文件
            for __blf_url in tqdm(
                self.blf_urls, desc=f"Processing BLF files for {__dbc_url}"
            ):
                self.read_single_can(
                    __dbc_url,
                    __dbc_data,
                    __blf_url,
                    "blf",
                    signal_names,
                    signal_corr,
                    step,
                    save_dir,
                    save_formats,
                )

            # 处理 ASC 文件
            for __asc_url in tqdm(
                self.asc_urls, desc=f"Processing ASC files for {__dbc_url}"
            ):
                self.read_single_can(
                    __dbc_url,
                    __dbc_data,
                    __asc_url,
                    "asc",
                    signal_names,
                    signal_corr,
                    step,
                    save_dir,
                    save_formats,
                )

    def read_can_files_multi(
        self,
        signal_names: Optional[List[str]] = None,
        signal_corr: Optional[Dict[str, str]] = None,
        step: float = 0.002,
        save_dir: str = r"./can_decoded",
        save_formats: Tuple[str, ...] = (".csv", ".parquet", ".mat"),
    ) -> None:
        """
        Read multiple CAN files and decode them using the provided DBC data (multi-process).

        Args:
            dbc_input (Union[str, List[str]]): Path(s) to the DBC file(s).
            can_input (Union[str, List[str]]): Path(s) to the CAN file(s).
            signal_names (Optional[List[str]]): List of signal names to decode.
            signal_corr (Optional[Dict[str, str]]): Signal name corrections.
            step (float): Raster step size.
            save_dir (str): Directory to save the output files.
            save_formats (Tuple[str, ...]): File formats to save (e.g., ".csv", ".parquet", ".mat").
        """

        # 确保保存目录存在
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # 构建任务列表
        tasks = []
        for __dbc_url, __dbc_data in self.dbcs:
            for __blf_url in self.blf_urls:
                tasks.append((__dbc_url, __dbc_data, __blf_url, "blf"))
            for __asc_url in self.asc_urls:
                tasks.append((__dbc_url, __dbc_data, __asc_url, "asc"))

        with Pool() as pool:
            list(
                tqdm(
                    pool.starmap(
                        self.read_single_can,
                        [
                            (
                                dbc_url,
                                dbc_data,
                                log_file_path,
                                file_type,
                                signal_names,
                                signal_corr,
                                step,
                                save_dir,
                                save_formats,
                            )
                            for dbc_url, dbc_data, log_file_path, file_type in tasks
                        ],
                    ),
                    total=len(tasks),
                    desc="Processing CAN files",
                )
            )

if __name__ == "__main__":
    can_decoder = CanDecoder(
        "/home/p30021181206/uk.dbc",
        "/home/p30021181206/Pictures/share_folder/20250430_SZ_UKEB",
    )
    can_decoder.read_can_files(
        save_dir="/home/p30021181206/Documents/data/",
        save_formats=(
            ".csv",
            ".parquet",
        ),
    )