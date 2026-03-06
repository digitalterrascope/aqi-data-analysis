import sys
import os

# Add project root to Python path
sys.path.append(os.path.abspath(".."))

# Now this works
import aqi_pkg as ap
import utils as ut

from datetime import datetime, timedelta
from collections import deque, defaultdict, Counter

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_palette("colorblind")
plt.style.use("tableau-colorblind10")
import pandas as pd
import polars as pl
import numpy as np

from sqlalchemy.dialects.mysql import insert
from tqdm import tqdm

from aqi_pkg.db import get_session
from aqi_pkg.tables import Entry, MetricAverages, IsDuplicate, UnitConversions
from aqi_pkg.filters import *
from aqi_pkg.data_scripts.create_subindicies import *

import threading