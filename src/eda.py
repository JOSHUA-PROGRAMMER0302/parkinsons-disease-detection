import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# EDA helpers will go here


def statistical_summary(df: pd.DataFrame, include_columns: list | None = None, exclude_columns: list | None = None) -> pd.DataFrame:
	"""Return mean, std, min and max for all numerical features.

	Args:
		df: Input DataFrame.
		include_columns: Optional list of numerical columns to include (default: all numeric).
		exclude_columns: Optional list of columns to exclude from the numeric selection.

	Returns:
		A DataFrame indexed by feature with columns ['mean', 'std', 'min', 'max'].
	"""
	num_df = df.select_dtypes(include=[np.number]).copy()

	if include_columns is not None:
		num_df = num_df.loc[:, [c for c in include_columns if c in num_df.columns]]

	if exclude_columns:
		num_df = num_df.drop(columns=[c for c in exclude_columns if c in num_df.columns], errors='ignore')

	stats = pd.DataFrame({
		'mean': num_df.mean(),
		'std': num_df.std(),
		'min': num_df.min(),
		'max': num_df.max(),
	})

	# Nicely format numeric output
	stats = stats.round(6)

	print("Statistical summary (mean, std, min, max):")
	print(stats)

	return stats


def plot_correlation_heatmap(df: pd.DataFrame,
							 figsize: tuple = (15, 12),
							 annot: bool = True,
							 cmap: str = "coolwarm",
							 fmt: str = ".2f",
							 save_path: Optional[str] = None) -> pd.DataFrame:
	"""Plot a correlation heatmap for numerical features using seaborn.

	Args:
		df: Input DataFrame.
		figsize: Figure size tuple, default (15, 12).
		annot: Whether to annotate cells with correlation values.
		cmap: Colormap for the heatmap.
		fmt: Annotation format string.
		save_path: Optional path to save the figure (PNG). If None, the figure is not saved.

	Returns:
		The correlation DataFrame that was plotted.
	"""
	num_df = df.select_dtypes(include=[np.number]).copy()
	corr = num_df.corr()

	plt.figure(figsize=figsize)
	sns.heatmap(corr, annot=annot, fmt=fmt, cmap=cmap, cbar=True, square=False)
	plt.title("Feature Correlation Heatmap")
	plt.tight_layout()

	if save_path:
		plt.savefig(save_path, dpi=300, bbox_inches='tight')

	plt.show()
	return corr


def plot_top10_boxplots(df: pd.DataFrame, target: Optional[str] = None, method: str = "correlation",
						top_k: int = 10, figsize: tuple = (16, 12), palette: str = "Set2",
						orient: str = "v", save_path: Optional[str] = None):
	"""Plot box plots for top features comparing classes (Parkinson's vs Healthy).

	Selection methods:
	  - 'correlation': select top features by absolute Pearson correlation with target (requires numeric target)
	  - 'variance': select top features by variance

	Args:
		df: DataFrame containing features and target.
		target: target column name. If None, will search for common names ('status','target','label').
		method: 'correlation' or 'variance'.
		top_k: number of top features to plot (default 10).
		figsize: matplotlib figure size.
		palette: seaborn palette for boxplots.
		orient: 'v' or 'h' orientation for boxplots.
		save_path: optional path to save the figure.

	Returns:
		List of selected feature names and the matplotlib Figure, Axes.
	"""
	# Identify target column
	tgt_col = target
	if tgt_col is None:
		for cand in ("status", "target", "label"):
			if cand in df.columns:
				tgt_col = cand
				break

	if tgt_col is None or tgt_col not in df.columns:
		raise ValueError("Target column not found. Provide `target` or include a 'status/target/label' column.")

	# Ensure target is categorical
	y = df[tgt_col]

	# Prepare numeric features (exclude target if numeric)
	num_df = df.select_dtypes(include=[np.number]).copy()
	if tgt_col in num_df.columns:
		num_df = num_df.drop(columns=[tgt_col])

	if num_df.shape[1] == 0:
		raise ValueError("No numeric features available for boxplots.")

	# Select top features
	if method == "correlation":
		# require numeric target
		if not np.issubdtype(y.dtype, np.number):
			# try to convert to numeric labels
			y_encoded = pd.factorize(y)[0]
		else:
			y_encoded = y

		corrs = num_df.corrwith(pd.Series(y_encoded).astype(float)).abs()
		top_features = corrs.sort_values(ascending=False).head(top_k).index.tolist()
	elif method == "variance":
		variances = num_df.var()
		top_features = variances.sort_values(ascending=False).head(top_k).index.tolist()
	else:
		raise ValueError("`method` must be 'correlation' or 'variance'")

	# Plot
	n_feats = len(top_features)
	ncols = 2
	nrows = int(np.ceil(n_feats / ncols))
	fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
	axes = np.array(axes).reshape(-1)

	for i, feat in enumerate(top_features):
		ax = axes[i]
		sns.boxplot(x=tgt_col if orient == 'v' else feat,
					y=feat if orient == 'v' else tgt_col,
					data=df[[feat, tgt_col]], palette=palette, ax=ax)
		ax.set_title(feat)

	for j in range(n_feats, len(axes)):
		axes[j].axis('off')

	plt.tight_layout()
	if save_path:
		plt.savefig(save_path, dpi=300, bbox_inches='tight')
	plt.show()

	return top_features, fig, axes


def plot_top10_correlated_features(df: pd.DataFrame, target: Optional[str] = None,
								   top_k: int = 10, figsize: tuple = (10, 6), cmap: str = 'viridis',
								   save_path: Optional[str] = None) -> pd.DataFrame:
	"""Identify top K features most correlated with target and plot a horizontal bar chart.

	Args:
		df: DataFrame containing features and target.
		target: Name of the target column. If None, looks for 'status','target','label'.
		top_k: Number of top features to show (default 10).
		figsize: Figure size for the plot.
		cmap: Matplotlib colormap name to color bars by absolute correlation.
		save_path: Optional path to save the figure.

	Returns:
		A DataFrame with columns ['correlation'] for the selected features (signed correlation).
	"""
	# find target column
	tgt_col = target
	if tgt_col is None:
		for cand in ("status", "target", "label"):
			if cand in df.columns:
				tgt_col = cand
				break

	if tgt_col is None or tgt_col not in df.columns:
		raise ValueError("Target column not found. Provide `target` or include a 'status/target/label' column.")

	y = df[tgt_col]

	# numeric features
	num_df = df.select_dtypes(include=[np.number]).copy()
	if tgt_col in num_df.columns:
		num_df = num_df.drop(columns=[tgt_col])

	if num_df.shape[1] == 0:
		raise ValueError("No numeric features available to compute correlations.")

	# encode target if categorical
	if not np.issubdtype(y.dtype, np.number):
		y_enc = pd.factorize(y)[0]
	else:
		y_enc = y

	corrs = num_df.corrwith(pd.Series(y_enc).astype(float))
	corr_df = corrs.rename('correlation').to_frame()

	# select top_k by absolute correlation
	top = corr_df.reindex(corr_df['correlation'].abs().sort_values(ascending=False).head(top_k).index)

	# Plot horizontal bar chart
	vals = top['correlation']
	features = top.index.tolist()

	# color by absolute value
	abs_vals = vals.abs().values
	if abs_vals.max() > 0:
		norm = abs_vals / abs_vals.max()
	else:
		norm = abs_vals
	cmap_fn = plt.cm.get_cmap(cmap)
	colors = [cmap_fn(v) for v in norm]

	fig, ax = plt.subplots(figsize=figsize)
	y_pos = np.arange(len(features))
	ax.barh(y_pos, vals.values, color=colors)
	ax.set_yticks(y_pos)
	ax.set_yticklabels(features)
	ax.invert_yaxis()
	ax.set_xlabel('Correlation with target')
	ax.set_title(f'Top {len(features)} features correlated with {tgt_col}')

	# annotate values
	for i, v in enumerate(vals.values):
		ax.text(v + (0.01 if v >= 0 else -0.01), i, f'{v:.3f}', va='center', ha='left' if v >= 0 else 'right')

	plt.tight_layout()
	if save_path:
		plt.savefig(save_path, dpi=300, bbox_inches='tight')
	plt.show()

	return top

