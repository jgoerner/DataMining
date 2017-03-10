import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

class DataFramePreprocessor(object):
    """ Preprocessor to normalize continuous and encode categorical values
    """
        
    def __init__(self, encode=True, normalize=False):
        """ Initialize new object
        
        INPUT:
        ------
        encode:     Flag to enable one-hot-encoding, default=True
        normalize:  Flag to enable normalization, default=False
        
        
        OUTPUT:
        -------
        New DataFramePreprocessor object
        
        
        DESCRIPTION:
        ------------
        Based on the encode & normalization flag the DataFramePreprocessor will
        - one-hot-encode categorical columns
        - normalize continuous columns
        """
        self.normalize = normalize
        self.encode = encode
        self.label_encoder = None
        self.scaler = None
        self.cat_columns = None
        self.con_columns = None
        
    
    def fit_transform_label(self, label):
        """ Encode and transform label
        
        INPUT:
        ------
        label:
        
        OUTPUT:
        -------
        y:
        
        
        DESCRIPTION:
        ------------
        Based on the internal instance of sklearn.preprocessing.LabelEncoder
        the class labels will be encoded and transformed.
        """
        self.label_encoder = LabelEncoder()
        dt_label = pd.DataFrame(self.label_encoder.fit_transform(label.values.ravel()),
                               index = label.index,
                               columns = label.columns)
        return dt_label
        
        
    def fit_transform(self, attributes, label=None):
        """ Learn preprocessing parameter and transform dataframe
        
        INPUT:
        ------
        attributes:  Input dataframe containing the attributes
        label:       Optional dataframe containing class labels
        
        
        OUTPUT:
        -------
        X:           Dataframe containing preprocessed attributes
        y:           Optional dataframe containing encoded class labels
        
        
        DESCRIPTION:
        ------------
        Based on the 'encode' and 'normalize' parameter, the method will
        
        - one-hot-encode categorical attributes
        - normalize continuous attributes
        - pass bool attributes without processing
        - encode labels
        
        and return the dataframe(s).
        """
        # Split label column from attributes if given
        if(label):
            dt_label = attributes.loc[:,label].to_frame()
            attributes = attributes.loc[:, attributes.columns != label]
        else:
            dt_label = None
            
        # Encode categorical columns if given
        dt_cat = attributes.select_dtypes(include=["O"])
        self.cat_columns = dt_cat.columns
        if self.encode:
            dt_cat = pd.get_dummies(dt_cat)
            
        # Normalize continuous colums if given
        dt_con = attributes.select_dtypes(include=["int64", "float64"])
        self.con_columns = dt_con.columns
        if self.normalize:
            self.scaler = StandardScaler()
            dt_con = pd.DataFrame(self.scaler.fit_transform(dt_con), 
                                       index=dt_con.index, 
                                       columns=dt_con.columns)
            
        # Pass bool colums
        dt_bool = attributes.select_dtypes(include=["bool"])
        
        # Encode label if given
        if label:
            #self.label_encoder = LabelEncoder()
            #dt_label[label] = self.label_encoder.fit_transform(dt_label.values.ravel())
            dt_label = self.fit_transform_label(dt_label)

        
        # Return merged dataframe + label (if given)
        return pd.concat([dt_cat, dt_con, dt_bool], axis=1), dt_label
    
    
    def inverse_transform_label(self, label):
        """ Invert encoded labels back to original class labels
        
        INPUT:
        ------
        label:   Dataframe containing encoded class labels
        
        
        OUTPUT:
        -------
        y:       Dataframe containing original class labels
        
        
        DESCRIPTION:
        ------------
        Based on the internal instance of sklearn.preprocessing.LabelEncoder
        the encoded class labels will be inverted and transformed.
        """
        return pd.DataFrame(self.label_encoder.inverse_transform(label),
                            index = label.index,
                            columns = label.columns)
    
    
    def inverse_transform(self, attributes, label=None):
        """ Invert, transform and merge dataframe(s) containing preprocessed attributes
        
        INPUT:
        ------
        attributes:  Dataframe containing the preprocessed attributes
        labels:      Dataframe containing the encoded labels
        
        
        OUTPUT:
        -------
        Xy:          Dataframe containing attributes (optionally labels) in original formats
        
        
        DESCRIPTION:
        ------------
        Based on the 'encode' and 'normalize' parameter, the method will
        
        - restore categorical attributes which have been one-hot-encoded
        - de-normalize continuous attributes to original interval ranges
        - pass bool attributes without processing
        - revert encode labels to original class labels
        
        and return the merged dataframe.
        """
        
        # Inverse continuous columns
        dt_con = attributes[self.con_columns]
        if self.normalize:
            dt_con = pd.DataFrame(self.scaler.inverse_transform(dt_con),
                                  index = dt_con.index,
                                  columns = dt_con.columns)
        
        # Inverse label dataframe
        dt_label = None
        if label is not None:
            dt_label = self.inverse_transform_label(label)
            
        # Pass bool columns
        dt_bool = attributes.select_dtypes(include=["bool"])
        
        # Inverse categorical columns - black magic following...
        dt_cat = None
        dt_cat_candidate = attributes[attributes.columns.difference(np.concatenate([dt_bool.columns,
                                                                                    dt_con.columns]))]
        for col in self.cat_columns:
            candidates = list(filter(lambda x: x.startswith(col), dt_cat_candidate.columns))
            values = list(map(lambda x: x.replace("{}_".format(col), ""), candidates))
            mapping = {idx : name for idx, name in enumerate(values)}
            aux_vector = np.arange(len(values))
            dt_cat =  pd.concat([dt_cat,
                                 pd.DataFrame((dt_cat_candidate.loc[:, candidates]*aux_vector).apply(np.sum, axis=1).map(mapping),
                                              index = dt_cat_candidate.loc[:, candidates].index,
                                              columns = [col]
                                              )], axis=1)
            
        # Return label dataframe
        return pd.concat([dt_cat, dt_con, dt_bool, dt_label], axis=1)