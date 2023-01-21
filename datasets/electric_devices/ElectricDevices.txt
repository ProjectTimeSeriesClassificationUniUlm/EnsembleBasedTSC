Problems

ElectricDevices LargeKitchenAppliances RefrigerationDevices
ScreenType SmallKitchenAppliances


Background:

These problems were taken from data recorded as part of government sponsored study called Powering the Nation. The intention was to collect behavioural data about how consumers use electricity within the home to help reduce the UK's carbon footprint. The data contains readings from 251 households, sampled in two-minute intervals over a month. The data required considerable preprocessing to get into a usable format.


We create two distinct types of problem: problems with similar usage patterns
(Refrigeration, Computers, Screen) and problems with dissimilar
usage patterns (Small Kitchen and Large Kitchen). The aim is that
problems with dissimilar usage patterns should be well suited to
time-domain classification, whilst those with similar consumption
patterns should be much harder.


The five problems we form are summarised in
Table~\ref{tab:electricityProblems}. Further information on these
data sets can be found at
https://sites.google.com/site/sdm2014timeseries/.

More on formatting from Jay.

\begin{table}[htbp]
	\centering
  	\caption{The five new TSC problems with class values}
  	\scriptsize
    \begin{tabular}{|c|c|}
    	\hline
    	Problem & Class Labels \\
    	\hline
    	Small Kitchen & Kettle, Microwave, Toaster \\
    	Large Kitchen & Dishwasher, Tumble Dryer, Washing Machine \\
    	Refrigeration & Fridge/Freezer, Refrigerator, Upright Freezer \\
    	Computers & Desktop, Laptop \\
    	Screen  & CRT TV, LCD TV, Computer Monitor \\
    	\hline
    \end{tabular}
  \label{tab:electricityProblems}
\end{table}
