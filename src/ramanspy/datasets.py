import os
import glob
from typing import List, Tuple
import pandas as pd
import scipy
import numpy as np
from io import BytesIO
from urllib.request import urlopen
from zipfile import ZipFile
import wget

from . import core
from . import load


def bacteria(dataset="train", folder=None) -> Tuple[core.SpectralContainer or list, np.ndarray or list]:
    """
    Raman spectra acquired from different bacterial and yeast isolates.

    >80k spectra across 30+ isolates. Ideal for classification modelling.

    Data from `Ho, CS. et al. (2019) <https://www.nature.com/articles/s41467-019-12898-9>`_.

    Must be downloaded first. Provided by authors on `DropBox <https://www.dropbox.com/sh/gmgduvzyl5tken6/AABtSWXWPjoUBkKyC2e7Ag6Da?dl=0>`_.

    Parameters
    ----------
    dataset : str, default='train'
        Which bacteria dataset to load.

        Available datasets are:

        - ``'train'`` - 60k spectra, 2k for each of 30 different reference bacterial and yeast isolates;
        - ``'val'`` - 3k spectra, 100 spectra for each of the reference isolates;
        - ``'test'`` - 3k spectra, 100 spectra for each of the reference isolates;
        - ``'clinical2018'`` - 12k spectra, 400 spectra for each of 30 patient isolates (distributed across 5 species);
        - ``'clinical2019'`` - 2.5k spectra, 100 spectra for each of 25 patient isolates (distributed across 5 species);
        - ``'labels'`` - The names of the species and antibiotics corresponding to the 30 classes.

    folder : str, default=None
        Path to the folder containing the downloaded data. If None, will use the root location. Irrelevant if ``dataset='labels'``.

    Returns
    -------
    SpectralContainer with spectral_data of shape (N, B)
        The Raman spectra provided in the selected dataset.
    np.ndarray[int] of shape (N, )
        The corresponding labels - indicating which bacteria species each data point corresponds to.


    References
    ----------
    Ho, CS., Jean, N., Hogan, C.A. et al. Rapid identification of pathogenic bacteria using Raman spectroscopy and deep learning. Nat Commun 10, 4927 (2019).


    Examples:
    ----------

    .. code::

        import ramanspy as rp
       
        # Load training and testing datasets
        X_train, y_train = rp.datasets.bacteria("train", path_to_data="path/to/data")
        X_test, y_test = rp.datasets.bacteria("test", path_to_data="path/to/data"))

        # Load the names of the species and antibiotics corresponding to the 30 classes
        y_labels, antibiotics_labels = rp.datasets.bacteria("labels")
    """
    _bacteria_splits = {
        "train": ['X_reference.npy', 'y_reference.npy'],
        "val": ['X_finetune.npy', 'y_finetune.npy'],
        "test": ['X_test.npy', 'y_test.npy'],
        "clinical2018": ['X_2018clinical.npy', 'y_2018clinical.npy'],
        "clinical2019": ['X_2019clinical.npy', 'y_2019clinical.npy'],
        "labels": None
    }
    if dataset not in _bacteria_splits:
        raise ValueError(
            f"{dataset} is not a valid split of the bacteria dataset. Available splits are {_bacteria_splits.keys()}.")

    if dataset == 'labels':
        y_labels = [
            'C. albicans', 'C. glabrata', 'K. aerogenes', 'E. coli 1', 'E. coli 2', 'E. faecium', 'E. faecalis 1',
            'E. faecalis 2', 'E. cloacae', 'K. pneumoniae 1', 'K. pneumoniae 2', 'P. mirabilis', 'P. aeruginosa 1',
            'P. aeruginosa 2', 'MSSA 1', 'MSSA 3', 'MRSA 1 (isogenic)', 'MRSA 2', 'MSSA 2', 'S. enterica',
            'S. epidermidis', 'S. lugdunensis', 'S. marcescens', 'S. pneumoniae 2', 'S. pneumoniae 1', 'S. sanguinis',
            'Group A Strep.', 'Group B Strep.', 'Group C Strep.', 'Group G Strep.']

        antibiotics_labels = [
            'Caspofungin', 'Caspofungin', 'Meropenem', 'Meropenem', 'Meropenem', 'Daptomycin', 'Penicillin', 'Penicillin',
            'Meropenem', 'Meropenem', 'Meropenem', 'Meropenem', 'TZP', 'TZP', 'Vancomycin', 'Vancomycin', 'Vancomycin',
            'Vancomycin', 'Vancomycin', 'Ciprofloxacin', 'Vancomycin', 'Vancomycin', 'Meropenem', 'Ceftriaxone',
            'Ceftriaxone', 'Penicillin', 'Penicillin', 'Penicillin', 'Penicillin', 'Penicillin']

        return y_labels, antibiotics_labels

    X_path, y_path = _bacteria_splits.get(dataset)

    X_data_path = X_path if folder is None else os.path.join(folder, X_path)
    y_data_path = y_path if folder is None else os.path.join(folder, y_path)
    wavenumbers_path = 'wavenumbers.npy' if folder is None else os.path.join(folder, 'wavenumbers.npy')

    X_data = np.load(X_data_path)
    y = np.load(y_data_path).astype(int)
    wavenumbers = np.load(wavenumbers_path)

    X = core.SpectralContainer(X_data, wavenumbers)

    return X, y


def covid19(file) -> Tuple[core.SpectralContainer, np.ndarray, np.ndarray]:
    """
    Raman spectra acquired from patients with COVID-19 and healthy controls.

    Data from `Yin G. et al. (2021) <https://pubmed.ncbi.nlm.nih.gov/33821082/>`_.

    Must be downloaded first. Available on `Kaggle <https://www.kaggle.com/datasets/sfran96/raman-spectroscopy-for-detecting-covid19>`_.

    Parameters
    ----------
    file : str, default=None
        Path to the file containing the downloaded data.

    Returns
    -------
    SpectralContainer with spectral_data of shape (N, B)
        The Raman spectra provided in the selected dataset.
    np.ndarray[int] of shape (N, )
        The corresponding labels - indicating which group each data point corresponds to.
    np.ndarray[string] of shape (B, )
        The names of the labels.

    References
    ----------
    Yin G, Li L, Lu S, Yin Y, Su Y, Zeng Y, Luo M, Ma M, Zhou H, Orlandini L, Yao D. An efficient primary screening of COVID‐19 by serum Raman spectroscopy. Journal of Raman Spectroscopy. 2021 May;52(5):949-58.

    Yin G, Li L, Lu S, Yin Y, Su Y, Zeng Y, Luo M, Ma M, Zhou H, Yao D, Liu G, Lang J. Data and code on serum Raman spectroscopy as an efficient primary screening of coronavirus disease in 2019 (COVID-19). figshare; 2020.


    Examples:
    ----------

    .. code::

        import ramanspy as rp

        # Load training dataset
        spectra, labels, label_names = rp.datasets.covid19(path_to_data="path/to/data")
    """
    df = pd.read_csv(file)

    labels, y = np.unique(df['diagnostic'].values, return_inverse=True)
    X_data = df.drop('diagnostic', axis=1).values
    wavenumbers = df.columns[:-1].values.astype(float)

    X = core.SpectralContainer(X_data, wavenumbers)

    return X, y, labels


def wheat_lines(file=None, download=True) -> Tuple[core.SpectralContainer, np.ndarray, np.ndarray]:
    """
    Raman spectra acquired from groups of wheat lines:

        - ``'COM'`` - Commercial cultivar;
        - ``'COM - 125mM'`` - Commercial cultivar treated with 125mM NaCl;
        - ``'ML1 - 125mM'`` - Mutant Line 1 treated with 125mM NaCl;
        - ``'ML2 - 125mM'`` - Mutant Line 2 treated with 125mM NaCl.

    Data from `ŞEN A. et al. (2023) <https://www.frontiersin.org/articles/10.3389/fpls.2023.1116876/full>`_.

    Available on `Zenodo <https://zenodo.org/record/7644521#.ZC7jV3bMK3A>`_.

    Can be downloaded directly through the function or downloaded separately. In the latter case, users just need to
    specifiy the location of the file to be loaded.

    Parameters
    ----------
    file : str, default=None
        Path to the file containing the downloaded data. Not used if ``download=True``.
    download : bool, default=True
        If ``True``, will download the data from Zenodo. Otherwise, will look for the data in the specified path given by ``path_to_data``.
        Note that if ``download=True``, data will be downloaded which may take some time.

    Returns
    -------
    SpectralContainer with spectral_data of shape (N, B)
        The Raman spectra provided in the selected dataset.
    np.ndarray[int] of shape (N, )
        The corresponding labels - indicating which group each data point corresponds to.
    np.ndarray[string] of shape (B, )
        The names of the labels.

    References
    ----------
    ŞEN A, Kecoglu I, Ahmed M, Parlatan U, Unlu M. Differentiation of advanced generation mutant wheat lines: Conventional techniques versus Raman spectroscopy. Frontiers in Plant Science. 2023;14.

    Examples:
    ----------

    .. code::

        import ramanspy as rp

        # Load training dataset
        spectra, labels, label_names = rp.datasets.wheat_lines()
    """
    if download:
        file = 'Data.mat'
        wget.download(f"https://zenodo.org/record/7644521/files/{file}?download=1")
        data = scipy.io.loadmat(file)
    else:
        data = scipy.io.loadmat(file)

    wavenumbers = data['Calx'].squeeze()

    labels = ['COM', 'COM_125mM', 'ML1_125mM', 'ML2_125mM']

    X_data = []
    y = []
    for i, dataset in enumerate(labels):
        X_data.append(data[dataset])
        y.append(np.repeat(i, data[dataset].shape[0]))

    X_data = np.concatenate(X_data)
    y = np.concatenate(y)

    X = core.SpectralContainer(X_data, wavenumbers)

    return X, y, labels


def adenine(file=None, download=True) -> Tuple[core.SpectralContainer, np.ndarray, np.ndarray]:
    """
    Raman spectra acquired from samples representing different levels of adenine concentrations.

    Data from `Fornasaro, Stefano, et al. (2020) <https://pubs.acs.org/doi/10.1021/acs.analchem.9b05658>`_.

    Can be downloaded directly through the function or downloaded separately. In the latter case, users just need to
    specifiy the location of the file to be loaded.

    Available on `Zenodo <https://zenodo.org/record/3572359#.ZGxwOxbMLDs>`_.

    Parameters
    ----------
    file : str, default=None
        Path to the file containing the downloaded data. Not used if ``download=True``.
    download : bool, default=True
        If ``True``, will download the data from Zenodo. Otherwise, will look for the data in the specified path given by ``path_to_data``.
        Note that if ``download=True``, data will be downloaded which may take some time.

    Returns
    -------
    SpectralContainer with spectral_data of shape (N, B)
        The Raman spectra provided in the selected dataset.
    pandas.DataFrame of shape (N, 8)
        8 additional features indicating sample collection parameters..
    np.ndarray[int] of shape (N, )
        The corresponding labels - indicating the adenine concentration which each data point corresponds to.


    References
    ----------
    Fornasaro S, Alsamad F, Baia M, Batista de Carvalho LA, Beleites C, Byrne HJ, Chiadò A, Chis M, Chisanga M, Daniel A, Dybas J. Surface enhanced Raman spectroscopy for quantitative analysis: results of a large-scale European multi-instrument interlaboratory study. Analytical chemistry. 2020 Feb 11;92(5):4053-64.

    Examples:
    ----------

    .. code::

        import ramanspy as rp

        # Load dataset
        spectra, additional_features, labels = rp.datasets.adenine()
    """
    if download:
        file = 'ILSdata.csv'
        wget.download(f"https://zenodo.org/record/3572359/files/{file}?download=1")
        df = pd.read_csv(file)
    else:
        df = pd.read_csv(file)

    y = df.pop('conc')
    X = df.loc[:, '400':]
    features = df.loc[:, :'replica']

    return X, features, y


def volumetric_cells(cell_type='THP-1', folder=None) -> List[core.SpectralVolume]:
    """
    A single volumetric scan of hiPSC cells.

    Data from `Kallepitis et al. (2017) <https://www.nature.com/articles/ncomms14843>`_.

    Must be downloaded first. Provided by authors on `Zenodo <https://zenodo.org/record/256329#.Y7wpc3bP1D_>`_.

    Parameters
    ----------
    cell_type : str, default='THP-1'
        The cell type to load. Supported cell types are:

        - ``'THP-1'`` - THP-1 cells (n=4);

    folder : str, default=None
        Path to the folder containing the data. If ``None``, will use the root location.


    Returns
    ---------
    list[SpectralVolume]
        A collection of volumetric data of the given cell type.

    References
    ----------
    Kallepitis, C., Bergholt, M., Mazo, M. et al. Quantitative volumetric Raman imaging of three dimensional cell cultures. Nat Commun 8, 14843 (2017).


    Examples:
    ----------

    .. code::

        import ramanspy as rp

        cells_volume = rp.datasets.volumetric_cells(cell_type='THP-1', path_to_data="path/to/data")
    """

    dir_ = cell_type if folder is None else os.path.join(folder, cell_type)

    volumes = [
        core.SpectralVolume.from_image_stack(
            [load.witec(matlab_file) for matlab_file in sorted(glob.glob(os.path.join(dir_, '*.mat'))) if replicate in matlab_file]
        ) for replicate in ['001', '002', '003', '004']
    ]

    return volumes


def MDA_MB_231_cells(dataset="train", folder=None) -> Tuple[core.SpectralContainer, core.SpectralContainer]:
    """
    170k pairs of low- and high-SNR data. I

    Ideal for developing and validating denoising models and algorithms.

    Data from `Horgan, C.C. et al. (2021) <https://pubs.acs.org/doi/full/10.1021/acs.analchem.1c02178>`_.

    Must be downloaded first. Provided by authors on `Google Drive <https://drive.google.com/drive/folders/1590Zqr56txK5_hVlrfe7oEIdcKoUTEIH>`_.

    All data has spectral dimensionality of 500, in the range (500, 1800) cm:sup:`-1`.


    Parameters
    ----------
    dataset : str, default='train'
        Which bacteria dataset to load.

        Available datasets are:

        - ``'train'`` - Just under 160k spectra.
        - ``'test'`` - Just under 13k spectra.

    folder : str, default=None
        Path to the folder containing the downloaded data. If None, will use the root location. Irrelevant if ``dataset='labels'``.


    Returns
    ---------
    SpectralContainer
        Low SNR input.
    SpectralContainer
        The corresponding high SNR target output.


    References
    ----------
    Horgan, C.C., Jensen, M., Nagelkerke, A., St-Pierre, J.P., Vercauteren, T., Stevens, M.M. and Bergholt, M.S., 2021. High-Throughput Molecular Imaging via Deep-Learning-Enabled Raman Spectroscopy. Analytical Chemistry, 93(48), pp.15850-15860.


    Examples:
    ----------

    .. code::

        import ramanspy as rp

        input, output = rp.datasets.MDA_MB_231_cells(path_to_data="path/to/data")
    """
    _splits = {
        "train": ['Train_Inputs', 'Train_Outputs'],
        "test": ['Test_Inputs', 'Test_Outputs'],
    }

    if dataset not in _splits:
        raise ValueError(
            f"{dataset} is not a valid split of the bacteria dataset. Available splits are {_splits.keys()}.")

    X_path, y_path = _splits.get(dataset)

    X_data_path = f"{X_path}.mat" if folder is None else os.path.join(folder, f"{X_path}.mat")
    y_data_path = f"{y_path}.mat" if folder is None else os.path.join(folder, f"{y_path}.mat")
    wavenumbers_path = 'axis.txt' if folder is None else os.path.join(folder, 'axis.txt')

    wavenumbers = np.loadtxt(wavenumbers_path)

    X = scipy.io.loadmat(X_data_path)
    X = core.SpectralContainer(X[X_path], wavenumbers)

    Y = scipy.io.loadmat(y_data_path)
    Y = core.SpectralContainer(Y[y_path], wavenumbers)

    return X, Y


def rruff(dataset: str, folder=None, download: bool = True) -> Tuple[List[core.SpectralContainer], List[dict]]:
    """
    Raman spectra acquired from various minerals.

    Data from the `RRUFF database <https://rruff.info/>`_.

    Can be downloaded directly through the function or downloaded separately. In the latter case, users just need to
    specifiy the location of the file to be loaded.

    Parameters
    ----------
    dataset : str
        The name of the RRUFF Raman dataset to load. Check available datasets `here <https://rruff.info/zipped_data_files/raman/>`_.
    folder : str, default=None
        Path to the folder containing the downloaded data. If None, will use the root location. Irrelevant if ``download=True`.
    download : bool, optional, default=True
        Whether to download the specified dataset or load it from a local directory. If ``download=False``, all .txt files
        from the directory provided via `dataset` will be loaded.


    Returns
    -------
    list[Spectrum]
        The Raman spectra provided.
    list[dict]
        List of metadata dictionaries, extracted from the header of the RRUFF data file.


    References
    ----------
    Lafuente B, Downs R T, Yang H, Stone N (2015) The power of databases: the RRUFF project. In: Highlights in Mineralogical Crystallography, T Armbruster and R M Danisi, eds. Berlin, Germany, W. De Gruyter, pp 1-30.


    Examples:
    ----------
    
    .. code:: 
    
        import ramanspy as rp
       
        # downloaded from the Internet
        rp.datasets.rruff('fair_oriented')
       
        # loaded from the given folder
        rp.datasets.rruff('path/to/dataset/folder/fair_oriented', download=False)
    """
    if download:
        return _download_rruff(dataset)
    else:
        dir_ = dataset if folder is None else os.path.join(folder, dataset)
        files = [open(filename, 'r') for filename in glob.glob(f"{dir_}/*.txt")]
        return _parse_rruff_files(files)


def _download_rruff(dataset):
    if not dataset.endswith('.zip'):
        dataset += ".zip"

    zipurl = f"https://rruff.info/zipped_data_files/raman/{dataset}"

    with urlopen(zipurl) as zipresp:
        with ZipFile(BytesIO(zipresp.read())) as zipfile:
            files = [zipfile.open(file_name) for file_name in zipfile.namelist()]

            return _parse_rruff_files(files)


def _parse_rruff_file(file) -> Tuple[List[core.SpectralContainer], dict] or None:
    metadata = {}
    intensities = []
    shift_axis = []

    for line in file.readlines():
        line = str(line, "utf-8") if not isinstance(line, str) else line

        if line == '\n':
            continue
        elif line.startswith("##"):
            ind = line.find('=')
            property_ = line[:ind].strip()
            data = line[ind + 1:].strip()

            metadata[property_] = data
        else:
            axis_value, intensity_value = line.split(",")

            intensities.append(float(intensity_value.strip()))
            shift_axis.append(float(axis_value.strip()))

    file.close()

    if len(intensities) > 0:
        return core.Spectrum(intensities, shift_axis), metadata


def _parse_rruff_files(files) -> Tuple[List[core.SpectralContainer], List[dict]]:
    spectra = []
    metadata = []
    for file in files:
        result = _parse_rruff_file(file)

        if result is None:
            continue

        spectrum, metadatum = result

        spectra.append(spectrum)
        metadata.append(metadatum)

    return spectra, metadata
