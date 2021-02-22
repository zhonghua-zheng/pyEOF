import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from .util.advanced_pca import CustomPCA

def get_time_space(df, time_dim, lumped_space_dims):
    """
    Get a DataFrame with the dim: [row (e.g.,time), column (e.g, space)]

    Parameters
    ----------

    df : A `pandas.core.frame.DataFrame`, e.g., 
        | time | lat | lon | PM2.5 |
        | 2011 | 66  | 88  | 22    |
        | 2011 | 66  | 99  | 23    |
        | 2012 | 66  | 88  | 24    |
        | 2012 | 66  | 99  | 25    |
        
 
    time_dim: time dim name (str), e.g., "time"
    
    lumped_space_dims: a list of the lumped space, e.g., ["lat","lon"]

    Returns
    -------
    dataframe with [row (e.g.,time), column (e.g, space)], e.g.,
        | time | loc1    | loc2    |
        |      | (66,88) | (66,99) |
        | 2011 | 22      | 24      |
        | 2012 | 23      | 25      |
    
    """
    return df.set_index([time_dim]+lumped_space_dims).unstack(lumped_space_dims)

def _scale(data):
    scaler = StandardScaler()
    X = scaler.fit_transform(data)
    return X, scaler

class df_eof(object):
    """EOF analysis for Pandas DataFrame"""
    
    def __init__(self, df, pca_type=None, n_components=None, scaler=True):
        """
        Create an EOF object.

        Parameters
        ----------

        df : A `pandas.core.frame.DataFrame` 
            dim: [row (e.g.,time), column (e.g, space)]
    
        Returns
        -------
        pca : A `df_eof` instance.

        """
        if scaler is True:
            self._data, self._scaler = _scale(df.values)  # get matrix and scaler
        else:
            self._data = df.values
        self._df_cols = df.columns  # get the column labels 
        self._df_index = df.index  # get the index of 
        
        (self._time, self._space) = self._data.shape # get the shape of the matrix
        placeholder_idx = np.argwhere(~np.isnan(df.values[0])).reshape(-1,) # find the non-missing indices as placeholders
        data_dropna = self._data[:, placeholder_idx] 
        
        if pca_type=="varimax":
            skpca = CustomPCA(n_components=n_components, 
                              rotation='varimax')
            skpca.fit(data_dropna)
            self._EOFs = np.full([n_components, self._space], np.nan)
            
        elif pca_type is None:
            skpca = PCA()
            skpca.fit(data_dropna)
            if (self._time <= self._space):
                self._EOFs = np.full([self._time, self._space], np.nan)
            else:
                self._EOFs = np.full([len(placeholder_idx), self._space], np.nan)

        else:
            skpca = pca_type
            skpca.fit(data_dropna)
            if (self._time <= self._space):
                self._EOFs = np.full([self._time, self._space], np.nan)
            else:
                self._EOFs = np.full([len(placeholder_idx), self._space], np.nan)
            
        self._PCs = skpca.transform(data_dropna) # Principal components
        self._EOFs[:, placeholder_idx]  = skpca.components_ # EOF (eigenvectors) with missing data
        self._eigvals = skpca.explained_variance_ # n_components largest eigenvalues 
        self._evf = skpca.explained_variance_ratio_ # Percentage of variance explained by each of the selected components.
        
    def pcs(self, s=0, n=1, prefix="PC"):
        """
        Get principal component time series (PCs).

        Parameters
        ----------
        s : Scaling
            0 : Un-scaled PCs (default).
            1 : PCs are divided by the square-root 
                of the eigenvalue
            2 : PCs are multiplied by the square-root 
                of the eigenvalue.

        n :
            Number of PCs to retrieve.

        Returns
        -------
        pcs: A dataframe [time, PCs]

        """

        n_str = [prefix+str(pc+1) for pc in range(n)]
            
        if s == 0:
            # Do not scale.
            return pd.DataFrame(data = self._PCs[:, 0:n].copy(),
                                index = self._df_index,
                                columns = n_str)
        elif s == 1:
            # Divide by the square-root of the eigenvalue.
            return pd.DataFrame(data = self._PCs[:, 0:n] / np.sqrt(self._eigvals[0:n]),
                                index = self._df_index,
                                columns = n_str)
        elif s == 2:
            # Multiply by the square root of the eigenvalue.
            return pd.DataFrame(self._PCs[:, 0:n] * np.sqrt(self._eigvals[0:n]),
                                index = self._df_index,
                                columns = n_str)
        else:
            raise ValueError('Scaling option should be "0", "1" or "2".')
            
    def eofs(self, s=0, n=1, prefix="EOF"):
        """
        Get Empirical Orthogonal Functions (EOFs).

        Parameters
        ----------
        s : Scaling
            0 : Un-scaled PCs (default).
            1 : PCs are divided by the square-root 
                of the eigenvalue
            2 : PCs are multiplied by the square-root 
                of the eigenvalue.

        n :
            Number of EOFs to retrieve.

        Returns
        -------
        eofs : A dataframe [PCs, space]

        """
        
        n_idx = pd.Index([eof_idx+1 for eof_idx in range(n)], name=prefix)
        
        if s == 0:
            # Do not scale.
            return pd.DataFrame(data = self._EOFs[0:n],
                                index = n_idx,
                                columns = self._df_cols)
        elif s == 1:
            # Divide by the square-root of the eigenvalues.
            return pd.DataFrame(data = self._EOFs[0:n] / np.sqrt(self._eigvals[0:n])[:, np.newaxis],
                                index = n_idx,
                                columns = self._df_cols)
        elif s == 2:
            # Multiply by the square-root of the eigenvalues.
            return pd.DataFrame(data = self._EOFs[0:n] * np.sqrt(self._eigvals[0:n])[:, np.newaxis],
                                index = n_idx,
                                columns = self._df_cols)
        else:
            raise ValueError('Scaling option should be "0", "1" or "2".')
            
    def eigvals(self, n=1):
        """
        Eigenvalues (decreasing variances) associated with each EOF.

        Parameters
        ----------
        n : Number of Eigenvalues to retrieve.

        Returns
        -------
        eigvals : A numpy array of Eigenvalue(s)

        """

        return self._eigvals[0:n].copy()
    
    def evf(self, n=1):
        """
        The amount of variance explained by each of the selected components (from 0 to 1)

        Parameters
        ----------
        n : Number of PCs to retrieve.

        Returns
        -------
        evf : A numpy array of variance explained by each of the selected components

        """

        return self._evf[0:n].copy()