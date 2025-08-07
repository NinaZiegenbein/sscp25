
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from scipy import stats
import os
CLINICAL_DATA_XL = "../ClinicalData_OUS_MADPatients_EIVIND_29_4_2021.xlsx"
PCA_SCORES_CSV = "./PCA_Results_final_height/pca_scores.csv"
NUM_MODES = 10
ANALYSIS_MODES = [f"M{i}" for i in range(1, NUM_MODES + 1)]


def load_patient_data(file_path):
    """Loads patient clinical data from an Excel file."""
    try:
        data = pd.read_excel(file_path).drop([0, 1])
        data["Pat_no"] = data["Pat_no"].astype(int)
        data.set_index("Pat_no", inplace=True)
        return data
    except Exception as e:
        print(f"Error loading file {file_path}: {e}")
        return None

def load_pca_scores(file_path):
    """Loads and standardizes PCA scores from a CSV file."""
    try:
        pca_scores = pd.read_csv(file_path).set_index("Pat_no")
        pca_scores.index = pca_scores.index.astype(int)
        pca_scores = (pca_scores - pca_scores.mean()) / pca_scores.std()
        return pca_scores
    except Exception as e:
        print(f"Error loading file {file_path}: {e}")
        return None

def calculate_mann_whitney(pca_event, pca_noevent, modes):
    """Calculates Mann-Whitney U test p-values for given modes."""
    return [stats.mannwhitneyu(pca_event[mode], pca_noevent[mode])[1] for mode in modes]

def annotate_p_values(ax, p_values, y_pos=3, color="black"):
    """Annotates p-values on the plot."""
    for i, p_val in enumerate(p_values):
        ax.annotate(f"P = {p_val:.2f}", (i - 0.4, y_pos), color=color)


clinical_data = load_patient_data(CLINICAL_DATA_XL)

pca_scores = load_pca_scores(PCA_SCORES_CSV)

if clinical_data is not None and pca_scores is not None:
   
    clinical_data["arrhythmic_composite"] = (
        clinical_data[["Aborted_cardiac_arrest", "Ventricular_tachycardia"]].fillna(0).sum(axis=1) > 0
    )
    clinical_data["any_arrhythmia"] = (
        clinical_data[["Aborted_cardiac_arrest", "Ventricular_tachycardia", "nsVT"]].fillna(0).sum(axis=1) > 0
    )

    
    pca_clinical = pca_scores.merge(clinical_data, on="Pat_no")
    
    pca_clinical_event = pca_clinical[pca_clinical["arrhythmic_composite"]]
    pca_clinical_noevent = pca_clinical[~pca_clinical["any_arrhythmia"]]
    
    any_arrhythmia = pca_clinical["any_arrhythmia"].sum()

    # Overlap between categories
    nsvt_aca_overlap = ((pca_clinical["nsVT"] == 1) & (pca_clinical["Aborted_cardiac_arrest"] == 1)).sum()
    vt_aca_overlap = ((pca_clinical["Ventricular_tachycardia"] == 1) & (pca_clinical["Aborted_cardiac_arrest"] == 1)).sum()
    vt_nsvt_overlap = ((pca_clinical["Ventricular_tachycardia"] == 1) & (pca_clinical["nsVT"] == 1)).sum()
    all_three = ((pca_clinical["nsVT"] == 1) & (pca_clinical["Aborted_cardiac_arrest"] == 1) & (pca_clinical["Ventricular_tachycardia"] == 1)).sum()

    print(f"nsVT and ACA overlap: {nsvt_aca_overlap}")
    print(f"VT and ACA overlap: {vt_aca_overlap}")
    print(f"VT and nsVT overlap: {vt_nsvt_overlap}")
    print(f"All three overlap: {all_three}")
    
    # Statistical Analysis
    mann_whitney_p_values = calculate_mann_whitney(
        pca_clinical_event, pca_clinical_noevent, ANALYSIS_MODES
    )
    print(pd.DataFrame([mann_whitney_p_values], columns=ANALYSIS_MODES, index=["p-value"]))

    # Prepare Data for Plotting
    pca_scores_stacked = pca_scores[ANALYSIS_MODES].stack().reset_index(level=1)
    pca_scores_stacked.columns = ["mode", "score"]
    clinical_data_pca_stacked = clinical_data.merge(pca_scores_stacked, on="Pat_no")

    # Plotting
    plt.rcParams.update({"font.size": 14})
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(
        data=clinical_data_pca_stacked,
        x="mode",
        y="score",
        hue="arrhythmic_composite",
        ax=ax,
        palette="Set2",
    )


    ax.set_ylim(-3, 5)
    annotate_p_values(ax, mann_whitney_p_values, y_pos=4)
    ax.set_xlabel("Patient Mode Score (Standardized)")
    ax.set_ylabel("PCA Score")
    # ax.legend(title="Arrhythmic Composite", loc="lower center")

    plt.tight_layout()
    if not os.path.exists("./PCA_Results_final_height/figures"):
        os.makedirs("./PCA_Results_final_height/figures")
    plt.savefig("./PCA_Results_final_height/figures/arrhythmia_mode_comparison.svg")
    plt.show()




################# ILR ####################
# loopdata = clinical_data.merge(pca_scores, on = "Pat_no").query("ILR_y_n == 1")
# nol =clinical_data.merge(pca_scores, on = "Pat_no").query("ILR_y_n == 0")

# loopdata["loop_nsVT"] = loopdata["NSVT_count"] > 0
# loop_nsVT = loopdata.loc[loopdata["loop_nsVT"]]
# loop_nonsVT = loopdata.loc[~loopdata["loop_nsVT"]]

# manu_stats_loop = [stats.mannwhitneyu(loop_nsVT[M], loop_nonsVT[M])[1] for M in ANALYSIS_MODES]
# df_ilr = pd.DataFrame([manu_stats_loop], columns = ANALYSIS_MODES, index = ["p-value"])
 # Plotting
# plt.rcParams.update({"font.size": 14})
# fig, ax = plt.subplots(figsize=(12, 6))
# sns.boxplot(
#     data=clinical_data_pca_stacked,
#     x="mode",
#     y="score",
#     hue="arrhythmic_composite",
#     ax=ax,
#     palette="Set2",
# )


# ax.set_ylim(-3, 5)
# annotate_p_values(ax, manu_stats_loop, y_pos=4)
# ax.set_xlabel("Patient Mode Score (Standardized)")
# ax.set_ylabel("PCA Score")
# ax.legend(title="ILR", loc="lower center")

# plt.tight_layout()
# # if not os.path.exists("./PCA_Results_final_height/figures"):
# #     os.makedirs("./PCA_Results_final_height/figures")
# # plt.savefig("./PCA_Results_final_height/figures/arrhythmia_mode_ilr.png")
# plt.show()
