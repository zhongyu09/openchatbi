"""
    model adtributor

    DEPRECATED: This module is marked for deprecation.
    Please use the new core algorithm in `openchatbi.analysis.adtributor` and the Tool wrapper `openchatbi.tool.adtributor_tool.adtributor_drilldown` instead.
"""
import logging

import numpy as np

from df_utils import NO_SPECIFIC_INV, ATTRS_NOT_FOUND


def additional_check(dimension, df, attr_list):
    # change name: valid_candidate

    # total_surprise = dim_elems.loc[attr_list, 'surprise'].sum()
    # total_predict = dim_elems.loc[attr_list, 'predict'].sum()
    # total_real = dim_elems.loc[attr_list, 'real'].sum()
    """
    1. if rc real value proportion ~= 100%, (find top 98% attrs and count about 98% requests), skip this dimension
    2. if all attrs evenly drop. skip this check for now due to results not stable
    """

    # check proportion ~= 100% case
    checks = {
        'proportion': 0.98,
        'base_proportion': 0.98
    }
    reason = []
    for check, threshold in checks.items():
        if check not in df.columns:
            continue
        target_proportion = df.loc[df[dimension].isin(attr_list)][check].sum()
        if target_proportion >= threshold:
            return False, f"skip: root cause items{attr_list} {check} is {target_proportion:.2%} ~ 100%"
        reason.append(f"{check} is {target_proportion:.2%} < {threshold:.2%}")

    return True, ". ".join(reason)


def add_surpise(df, derived, merged_divide=1):
    """
    Computes the surprise for all elements in the dataframe.
    :param df: pandas dataframe.
    :param derived: boolean, if derived measures are used.
    :param merged_divide: int, if the total sum should be divided
      (this is true if the dataframe elements have been merged over the dimensions
      as done in the adtributor code).
    :return: dataframe with added column for the surprise.
    """
    def compute_surprise(col_real, col_predict):
        with np.errstate(divide='ignore'):
            f = df[col_predict].sum() / merged_divide
            a = df[col_real].sum() / merged_divide

            p = df[col_predict] / f
            q = df[col_real] / a
            p_term = np.nan_to_num(p * np.log(2 * p / (p + q)))
            q_term = np.nan_to_num(q * np.log(2 * q / (p + q)))
            surprise = 0.5 * (p_term + q_term)
        return surprise

    if derived:
        df['surprise'] = compute_surprise('real_numerator', 'predict_numerator') + \
            compute_surprise('real_denominator', 'predict_denominator')
    else:
        df['surprise'] = compute_surprise('real', 'predict')
    return df


def add_explanatory_power(df, derived, issue_type='drop'):
    """
    Computes the explanatory power for all elements in the dataframe.
    :param df: pandas dataframe.
    :param derived: boolean, if derived measures are used.
    :return: pandas dataframe with added column for the explanatory power.
    """
    if derived:
        f_a = df['predict_numerator'].sum()
        f_b = df['predict_denominator'].sum()

        n = (df['real_numerator'] - df['predict_numerator']) * f_b - \
            (df['real_denominator'] - df['predict_denominator']) * f_a
        d = f_b * (f_b + df['real_denominator'] - df['predict_denominator'])
        df['ep'] = n / d

        # Normalize to sum up to 1
        df['ep'] = df['ep'] / df['ep'].sum()
    else:
        f = df['predict'].sum()
        a = df['real'].sum()

        df['ep'] = (df['real'] - df['predict']) / (a - f)
    return df


def merge_dimensions(df, derived):
    if derived:
        df['predict'] = df['predict_numerator'] / df['predict_denominator']
        df['real'] = df['real_numerator'] / df['real_denominator']
    # df['element'] = df[dimensions]
    df = df.reset_index(drop=True)
    return df


def adtributor(derived, df_dict, dimension_weights=None,
               tep=0.7, teep=0.02, k=1, issue_type='drop'):
    '''
    Analyzes the input data and identifies candidate dimensions for drill-down analysis.
    :param derived: bool, whether the input data is derived
    :param df_dict: dict, a dictionary containing dimension names as keys and corresponding dataframes as values
    :param dimension_weights: dict, keys: dimension names; values: weights - optional default weight is 1
    :param tep: float, threshold for cumulative explanatory power
    :param teep: float, threshold for individual explanatory power
    :param k: int, number of top candidate dimensions to return
    :param issue_type: str, type of issue to consider ('drop' or 'rise')
    example:
    df1 = pd.DataFrame({
        'site_section': [1, 2, 3], -- dimension
        'predict': [0.5, 0.1, 0.9],
        'real': [0.7, 0.9, 1.0],
        'proportion': [0.2, 0.5, 0.3],
        'base_proportion': [0.2, 0.5, 0.3] -- optional
    })
    df_dict = {
        'site_section': df1,
    }
    :return: tuple(result, ranked_dim, debug) 
        result: Dict[dim: List[elements]]
        ranked_dim: List[dim]
        debug: List[Dict[elements, ep, surprise]]
    '''
    reason_flag = ''
    if dimension_weights is None:
        dimension_weights = {}
    candidates = []
    ranked_dimensions = []
    for d, dim_df in df_dict.items():
        if issue_type == 'drop':
            dim_df = dim_df.loc[dim_df['predict'] > dim_df['real'] + 0.001]
        elif issue_type == 'rise':
            dim_df = dim_df.loc[dim_df['predict'] + 0.001 < dim_df['real']]
        if dim_df.empty:
            logging.info(f"Skip {d} drill-down: all attribute should {issue_type}")
            reason_flag = NO_SPECIFIC_INV
            continue
        elements = merge_dimensions(dim_df, derived)
        elements = add_explanatory_power(elements, derived, issue_type)
        # elements = add_deviation_score(elements)
        elements = add_surpise(elements, derived, merged_divide=len([d]))
        dim_elems = elements.set_index(d)
        dim_elems = dim_elems.sort_values('surprise', ascending=False)
        cumulative_ep = dim_elems.loc[dim_elems['ep'] > teep, 'ep'].cumsum()

        attr_list = cumulative_ep.index.values.tolist()
        dimension_weight = dimension_weights.get(d, 1)

        total_suprise = dim_elems.loc[:, 'surprise'].sum() * dimension_weight

        candidate = {
            'explanatory_power': cumulative_ep.max(),
            'tep': tep,
            'total_suprise': total_suprise,
            'dimension': d
        }
        reason = ''

        if np.any(cumulative_ep > tep):
            idx = (cumulative_ep > tep).idxmax()
            attr_list = cumulative_ep.loc[:idx].index.values.tolist()
            further_check_passed, reason = additional_check(d, dim_df, attr_list)
            if further_check_passed:
                candidate['elements'] = attr_list
                candidate['surprise'] = dim_elems.loc[attr_list, 'surprise'].sum() * dimension_weight
        else:
            reason = f"skip: cumulative_ep({cumulative_ep.max()}) < tep({tep})"
            reason_flag = ATTRS_NOT_FOUND
        candidate['reason'] = reason
        logging.info(f"Dimension: {d}, {reason}")
        candidates.append(candidate)

    ranked_dimensions = sorted(candidates, key=lambda x: x['total_suprise'], reverse=True)
    ranked_dimensions = [d['dimension'] for d in ranked_dimensions]

    rc_candidates = sorted([c for c in candidates if 'elements' in c], key=lambda t: t['surprise'], reverse=True)[:k]
    candidate_dict = {c['dimension']: c['elements'] for c in rc_candidates}

    return candidate_dict, ranked_dimensions, {d['dimension']: d for d in candidates}, reason_flag
