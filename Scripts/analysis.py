import jsonlines
import numpy as np
from radon.complexity import cc_visit
import ast
from scipy.stats import stats
import tiktoken
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.interpolate import make_interp_spline


def count_lines_of_code(code):
    lines = code.split('\n')
    code_lines = 0
    for line in lines:
        stripped_line = line.strip()
        if stripped_line and not stripped_line.startswith("#"):
            code_lines += 1
    return code_lines


def count_cyclomatic_complexity_of_code(code):
    blocks = cc_visit(code)
    complexity = [block.complexity for block in blocks]
    try:
        result = sum(complexity) / len(complexity)
    except:
        result = 1
    return result


def count_api_in_code(code):
    tree = ast.parse(code)
    return sum(isinstance(node, ast.Call) for node in ast.walk(tree))


def count_comments_in_code(code):
    comment_lines = 0
    for line in code.split("\n"):
        if "# " in line:
            comment_lines += 1
    return comment_lines


def count_token_in_description(description):
    enc = tiktoken.get_encoding("o200k_base")
    result = enc.encode(description)
    return len(result)


def get_canonical_solution_distribution(file):  # get the code feature distribution of canonical solution. Table 2
    line_of_code = []
    cc_of_code = []
    api_in_code = []
    token_in_description = []
    with jsonlines.open(file, "r") as reader:
        for obj in reader:
            description = obj["prompt"]
            if "humaneval" in file:
                code = description[:description.index(":\n") + 2] + obj["canonical_solution"]
            else:
                code = obj["canonical_solution"]
            line_of_code.append(count_lines_of_code(code))
            cc_of_code.append(count_cyclomatic_complexity_of_code(code))
            api_in_code.append(count_api_in_code(code))
            token_in_description.append(count_token_in_description(description))
    return line_of_code, cc_of_code, api_in_code, token_in_description


def get_code_feature_difference(
        file):  # get the difference of the code features between the correct code and the canonical solution
    with jsonlines.open(file, "r") as reader:
        line_of_code = []
        cc_of_code = []
        api_in_code = []
        for obj in reader:
            if obj["result"] == 1:
                description = obj["prompt"]
                code = obj["solution"]
                if "humaneval" in file:
                    canonical_solution = description + obj["canonical_solution"]
                else:
                    canonical_solution = obj["canonical_solution"]
                line_of_code.append(count_lines_of_code(code) - count_lines_of_code(canonical_solution))
                cc_of_code.append(
                    count_cyclomatic_complexity_of_code(code) - count_cyclomatic_complexity_of_code(
                        canonical_solution))
                api_in_code.append(count_api_in_code(code) - count_api_in_code(canonical_solution))
    return line_of_code, cc_of_code, api_in_code


def compare_code_feature(humaneval, mbpp,
                         apps):  # compare the code features of the correct code and the canonical solution
    humaneval_line_of_code = []
    humaneval_cc_of_code = []
    humaneval_api_in_code = []
    for file in humaneval:
        line_of_code, cc_of_code, api_in_code = get_code_feature_difference(file)
        humaneval_line_of_code.append(line_of_code)
        humaneval_cc_of_code.append(cc_of_code)
        humaneval_api_in_code.append(api_in_code)
    mbpp_line_of_code = []
    mbpp_cc_of_code = []
    mbpp_api_in_code = []
    for file in mbpp:
        line_of_code, cc_of_code, api_in_code = get_code_feature_difference(file)
        mbpp_line_of_code.append(line_of_code)
        mbpp_cc_of_code.append(cc_of_code)
        mbpp_api_in_code.append(api_in_code)
    apps_line_of_code = []
    apps_cc_of_code = []
    apps_api_in_code = []
    for file in apps:
        line_of_code, cc_of_code, api_in_code = get_code_feature_difference(file)
        apps_line_of_code.append(line_of_code)
        apps_cc_of_code.append(cc_of_code)
        apps_api_in_code.append(api_in_code)
    return humaneval_line_of_code, humaneval_cc_of_code, humaneval_api_in_code, mbpp_line_of_code, mbpp_cc_of_code, mbpp_api_in_code, apps_line_of_code, apps_cc_of_code, apps_api_in_code


def plot_area_chart(humaneval, mbpp, apps):  # plot the area chart of the code features. Figure 1.
    humaneval_line_of_code, humaneval_cc_of_code, humaneval_api_in_code, mbpp_line_of_code, mbpp_cc_of_code, mbpp_api_in_code, apps_line_of_code, apps_cc_of_code, apps_api_in_code = compare_code_feature(
        humaneval, mbpp, apps)
    plot_smooth_area_chart_3(humaneval_line_of_code, humaneval_cc_of_code, humaneval_api_in_code, name="humaneval")
    plot_smooth_area_chart_3(mbpp_line_of_code, mbpp_cc_of_code, mbpp_api_in_code, name="mbpp")
    plot_smooth_area_chart_3(apps_line_of_code, apps_cc_of_code, apps_api_in_code, name="apps")


def plot_smooth_area_chart_3(data_line_of_code, data_cc_of_code, data_api_in_code, name):
    def avg_interp_spline(x, y, interval=3, k=3, num=500):
        if len(x) > interval * 10:
            x_merged = x[::interval]
            y_merged = np.array([y[i:i + interval].mean() for i in range(0, len(y), interval)])
        else:
            x_merged = x
            y_merged = y

        while len(x_merged) < 2:
            interval *= 2
            x_merged = x[::interval]
            y_merged = np.array([y[i:i + interval].mean() for i in range(0, len(y), interval)])

        if len(x_merged) < 5:
            raise ValueError("Not enough data points to perform interpolation.")
        cspline = make_interp_spline(x_merged, y_merged, k=k)
        x_new = np.linspace(min(x_merged), max(x_merged), num)
        y_new = np.clip(cspline(x_new), 0, None)

        return x_new, y_new

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    datas = [data_line_of_code, data_cc_of_code, data_api_in_code]
    legends = ["Claude-3", "Deepseek-Coder", "GPT-3.5", "GPT-4", "Llama3", "Phi3", "StarCoder2"]

    for i in range(3):

        x_max = -1
        x_min = 1
        y_max = -1
        y_min = 1

        for index, loc in enumerate(datas[i]):
            frequency = pd.Series(loc).value_counts().sort_index()
            percentage = frequency / len(loc)

            x = np.array(percentage.index)
            y = percentage.values

            x_smooth, y_smooth = avg_interp_spline(x, y, interval=2, num=100)

            x_max = max(x_max, max(x_smooth))
            x_min = min(x_min, min(x_smooth))
            y_max = max(y_max, max(y_smooth))
            y_min = min(y_min, min(y_smooth))

            axs[i].plot(x_smooth, y_smooth, label=f'{legends[index]}')
            axs[i].fill_between(x_smooth, 0, y_smooth, alpha=0.3)

        rate = 1.2
        x_max *= rate
        x_min *= rate
        y_max *= rate
        y_min *= rate
        axs[i].set_xlim(x_min, x_max)
        axs[i].set_ylim(y_min, y_max)

    title_fontsize = 16
    label_fontsize = 16
    axis_fontsize = 14
    for j in range(3):
        axs[j].tick_params(axis='both', which='major', labelsize=axis_fontsize)

        axs[j].set_xlabel("Difference", fontsize=label_fontsize)
        axs[j].set_ylabel("Percentage", fontsize=label_fontsize)
    axs[0].set_title('Lines of Code', fontsize=title_fontsize)
    axs[1].set_title('CC of Code', fontsize=title_fontsize)
    axs[2].set_title('API in Code', fontsize=title_fontsize)

    axs[0].legend()
    plt.tight_layout()
    plt.savefig(f"{name}.pdf")
    plt.show()


def comment_distribution(humaneval, mbpp, apps, rwpb):  # count the number of comments in the correct code. Figure 2.
    humaneval_correct_comment = []
    humaneval_wrong_comment = []
    for file in humaneval:
        with jsonlines.open(file, "r") as reader:
            correct_comment = []
            wrong_comment = []
            for obj in reader:
                code = obj["solution"]
                if obj["result"] == 1:
                    correct_comment.append(count_comments_in_code(code))
                else:
                    wrong_comment.append(count_comments_in_code(code))
            humaneval_correct_comment.append(correct_comment)
            humaneval_wrong_comment.append(wrong_comment)
    humaneval_p_values = significant_test(humaneval_correct_comment, humaneval_wrong_comment)
    mbpp_correct_comment = []
    mbpp_wrong_comment = []
    for file in mbpp:
        with jsonlines.open(file, "r") as reader:
            correct_comment = []
            wrong_comment = []
            for obj in reader:
                code = obj["solution"]
                if obj["result"] == 1:
                    correct_comment.append(count_comments_in_code(code))
                else:
                    wrong_comment.append(count_comments_in_code(code))
            mbpp_correct_comment.append(correct_comment)
            mbpp_wrong_comment.append(wrong_comment)
    mbpp_values = significant_test(mbpp_correct_comment, mbpp_wrong_comment)
    apps_correct_comment = []
    apps_wrong_comment = []
    for file in apps:
        with jsonlines.open(file, "r") as reader:
            correct_comment = []
            wrong_comment = []
            for obj in reader:
                code = obj["solution"]
                if obj["result"] == 1:
                    correct_comment.append(count_comments_in_code(code))
                else:
                    wrong_comment.append(count_comments_in_code(code))
            apps_correct_comment.append(correct_comment)
            apps_wrong_comment.append(wrong_comment)
    apps_p_values = significant_test(apps_correct_comment, apps_wrong_comment)
    rwpb_correct_comment = []
    rwpb_wrong_comment = []
    for file in rwpb:
        with jsonlines.open(file, "r") as reader:
            correct_comment = []
            wrong_comment = []
            for obj in reader:
                code = obj["solution"]
                if obj["result"] == 1:
                    correct_comment.append(count_comments_in_code(code))
                else:
                    wrong_comment.append(count_comments_in_code(code))
            rwpb_correct_comment.append(correct_comment)
            rwpb_wrong_comment.append(wrong_comment)
    rwpb_p_values = significant_test(rwpb_correct_comment, rwpb_wrong_comment)
    print("Humaneval p-values: ", humaneval_p_values)
    print("MBPP p-values: ", mbpp_values)
    print("Apps p-values: ", apps_p_values)
    print("RWPB p-values: ", rwpb_p_values)
    # plot the results via boxplot
    labels = ["CL3", "DC", "GPT3.5", "GPT4", "LL3", "Phi3", "SC2"]
    fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(8, 12))
    fig.subplots_adjust(hspace=0.3)
    sns.boxplot(ax=axes[0, 0], data=humaneval_correct_comment, showfliers=False)
    axes[0, 0].set_title("Comment Line Nums in Correct Code")
    axes[0, 0].set_xticklabels(labels)

    sns.boxplot(ax=axes[0, 1], data=humaneval_wrong_comment, showfliers=False)
    axes[0, 1].set_title("Comment Line Nums in Incorrect Code")
    axes[0, 1].set_xticklabels(labels)

    sns.boxplot(ax=axes[1, 0], data=mbpp_correct_comment, showfliers=False)
    axes[1, 0].set_title("Comment Line Nums in Correct Code")
    axes[1, 0].set_xticklabels(labels)

    sns.boxplot(ax=axes[1, 1], data=mbpp_wrong_comment, showfliers=False)
    axes[1, 1].set_title("Comment Line Nums in Incorrect Code")
    axes[1, 1].set_xticklabels(labels)

    sns.boxplot(ax=axes[2, 0], data=apps_correct_comment, showfliers=False)
    axes[2, 0].set_title("Comment Line Nums in Correct Code")
    axes[2, 0].set_xticklabels(labels)

    sns.boxplot(ax=axes[2, 1], data=apps_wrong_comment, showfliers=False)
    axes[2, 1].set_title("Comment Line Nums in Incorrect Code")
    axes[2, 1].set_xticklabels(labels)

    sns.boxplot(ax=axes[3, 0], data=rwpb_correct_comment, showfliers=False)
    axes[3, 0].set_title("Comment Line Nums in Correct Code")
    axes[3, 0].set_xticklabels(labels)

    sns.boxplot(ax=axes[3, 1], data=rwpb_wrong_comment, showfliers=False)
    axes[3, 1].set_title("Comment Line Nums in Incorrect Code")
    axes[3, 1].set_xticklabels(labels)
    # 设置相同的纵坐标范围
    for ax in axes.flatten():
        ax.set_ylim(-1, 22)
    plt.savefig("boxplot.pdf", format="pdf")
    plt.show()


def significant_test(correct_comment, wrong_comment):
    p_values = []
    for c, w in zip(correct_comment, wrong_comment):
        t_stat, p_value = stats.ttest_ind(c, w)
        p_values.append(p_value)
    return p_values


def get_bug_type_distribution(humaneval, mbpp, apps, rwpb, bug_type):  # get the distribution of bug type. Figure 4.
    humaneval_bug = []
    for file in humaneval:
        with jsonlines.open(file, "r") as reader:
            for obj in reader:
                if obj["bug_type"].startswith(bug_type):
                    humaneval_bug.append(obj)
    mbpp_bug = []
    for file in mbpp:
        with jsonlines.open(file, "r") as reader:
            for obj in reader:
                if obj["bug_type"].startswith(bug_type):
                    mbpp_bug.append(obj)
    apps_bug = []
    for file in apps:
        with jsonlines.open(file, "r") as reader:
            for obj in reader:
                if obj["bug_type"].startswith(bug_type):
                    apps_bug.append(obj)
    rwpb_bug = []
    for file in rwpb:
        with jsonlines.open(file, "r") as reader:
            for obj in reader:
                if obj["bug_type"].startswith(bug_type):
                    rwpb_bug.append(obj)
    humaneval_bug_dict = {}
    for obj in humaneval_bug:
        if obj["bug_type"] in humaneval_bug_dict:
            humaneval_bug_dict[obj["bug_type"]] += 1
        else:
            humaneval_bug_dict[obj["bug_type"]] = 1
    mbpp_bug_dict = {}
    for obj in mbpp_bug:
        if obj["bug_type"] in mbpp_bug_dict:
            mbpp_bug_dict[obj["bug_type"]] += 1
        else:
            mbpp_bug_dict[obj["bug_type"]] = 1
    apps_bug_dict = {}
    for obj in apps_bug:
        if obj["bug_type"] in apps_bug_dict:
            apps_bug_dict[obj["bug_type"]] += 1
        else:
            apps_bug_dict[obj["bug_type"]] = 1
    rwpb_bug_dict = {}
    for obj in rwpb_bug:
        if obj["bug_type"] in rwpb_bug_dict:
            rwpb_bug_dict[obj["bug_type"]] += 1
        else:
            rwpb_bug_dict[obj["bug_type"]] = 1
    print("Humaneval: ", humaneval_bug_dict)
    print("MBPP: ", mbpp_bug_dict)
    print("Apps: ", apps_bug_dict)
    print("RWPB: ", rwpb_bug_dict)
    # generate a figure contain four pie charts
    fig, axes = plt.subplots(1, 4, figsize=(28, 7))
    fig.subplots_adjust(hspace=0.3, wspace=0.3)

    # 定义颜色方案
    color_palette = plt.cm.Reds([0.1, 0.3, 0.5, 0.7, 0.9])

    dicts = [humaneval_bug_dict, mbpp_bug_dict, apps_bug_dict, rwpb_bug_dict]
    titles = ["Humaneval", "MBPP", "Apps", "RWPB"]

    legend_labels = set()

    for ax, bug_dict, title in zip(axes.flat, dicts, titles):
        labels = list(bug_dict.keys())
        values = list(bug_dict.values())

        # 找到占比最小的两个部分
        min_two_indices = sorted(range(len(values)), key=lambda i: values[i])[:2]
        explode = [0.1 if i in min_two_indices else 0 for i in range(len(values))]
        wedges, texts, autotexts = ax.pie(values, labels=labels, autopct='%1.1f%%', colors=color_palette[:len(labels)],
                                          explode=explode)
        ax.set_title(title, fontsize=14)
        for text in texts + autotexts:
            text.set_fontsize(12)
        legend_labels.update(labels)

    sorted_legend_labels = sorted(legend_labels)
    handles = [plt.Line2D([0], [0], color=color_palette[i % len(color_palette)], lw=2) for i in
               range(len(sorted_legend_labels))]

    fig.legend(handles, sorted_legend_labels, loc='upper right', fontsize=12, title="Bug Types", title_fontsize=14)

    plt.show()
