import streamlit as st
import pandas as pd
import numpy as np

st.write(f"## Hello, World!")
a = st.slider("a")
x = np.linspace(-6, 6, 500)
df = pd.DataFrame(dict(y=np.sin(a*x)))
st.line_chart(df)