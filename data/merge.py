import pandas as pd

def read_csv_files():
    # Reading the node files directly with correct column names
    case_study_df = pd.read_csv("CaseStudy.csv")
    organ_df = pd.read_csv("Organ.csv")
    chemical_df = pd.read_csv("Chemical.csv")
    model_system_df = pd.read_csv("Model_system.csv")
    computational_model_df = pd.read_csv("Computational_model.csv")
    bioassay_df = pd.read_csv("Bioassay.csv") 
    experimental_condition_df = pd.read_csv("ExperimentalCondition.csv")
    measurable_endpoint_df = pd.read_csv("MeasurableEndpoint.csv")

    # Keep the same structure for the output by combining all node data
    final_nodes = pd.concat([
        case_study_df, organ_df, chemical_df, model_system_df, 
        computational_model_df, bioassay_df, experimental_condition_df, measurable_endpoint_df
    ], ignore_index=True)

    # Add necessary columns for output structure
    final_nodes["OrganName"] = final_nodes.get("OrganName", None)
    final_nodes["ChemicalName"] = final_nodes.get("ChemicalName", None)
    final_nodes["ChemicalCAS"] = final_nodes.get("ChemicalCAS", None)
    final_nodes["SMILES"] = final_nodes.get("SMILES", None) 
    final_nodes["InChIKey"] = final_nodes.get("InChIKey", None)            
    final_nodes["chemical_group"] = final_nodes.get("chemical_group", None) 
    final_nodes["ModelSystemName"] = final_nodes.get("ModelSystemName", None)
    final_nodes["ModelSystemCellType"] = final_nodes.get("ModelSystemCellType", None)
    final_nodes["ModelSystemDescription"] = final_nodes.get("ModelSystemDescription", None)
    final_nodes["ComputationalModelName"] = final_nodes.get("ComputationalModelName", None)
    final_nodes["ComputationalModelType"] = final_nodes.get("ComputationalModelType", None)
    final_nodes["ComputationalModelLanguage"] = final_nodes.get("ComputationalModelLanguage", None)
    final_nodes["ComputationalModelInput"] = final_nodes.get("ComputationalModelInput", None)
    final_nodes["ComputationalModelOutput"] = final_nodes.get("ComputationalModelOutput", None)
    final_nodes["BioassayName"] = final_nodes.get("BioassayName", None)   
    final_nodes["Measured"] = final_nodes.get("Measured", None)   
    final_nodes["exposure_duration"] = final_nodes.get("exposure_duration", None)
    final_nodes["exposure_concentration"] = final_nodes.get("exposure_concentration", None)
    final_nodes["condition_name"] = final_nodes.get("condition_name", None)
    final_nodes["ExperimentalConditionDescription"] = final_nodes.get("ExperimentalConditionDescription", None)
    final_nodes["MeasurableEndpointName"] = final_nodes.get("MeasurableEndpointName", None)
    final_nodes["MeasurableEndpointDescription"] = final_nodes.get("MeasurableEndpointDescription", None)
    final_nodes["MeasurableEndpointType"] = final_nodes.get("MeasurableEndpointType", None)
    final_nodes["_start"] = None
    final_nodes["_end"] = None
    final_nodes["_type"] = None

    return case_study_df, chemical_df, bioassay_df, model_system_df, computational_model_df, final_nodes

def create_edges(case_study_df, chemical_df, bioassay_df, model_system_df, computational_model_df):
    # Create an empty DataFrame to hold edges
    edges = pd.DataFrame(columns=[
        '_id', '_labels', 'CaseStudyName', 'CaseStudyDescription', 'OrganName', 
        'ChemicalName', 'ChemicalCAS', 'ModelSystemName', 'ModelSystemCellType', 
        'ModelSystemDescription', 'ComputationalModelName', 'ComputationalModelType', 
        'ComputationalModelLanguage', 'ComputationalModelInput', 'ComputationalModelOutput',
        'BioassayName', 'Measured', '_start', '_end', '_type'
    ])

    def split_and_create_edges(df, column_name, edge_type, start_column="_id"):
        """Helper function to split comma-separated values and create new edges"""
        expanded_edges = []
        for idx, row in df.iterrows():
            if pd.notna(row[column_name]):
                values = row[column_name].split(',')
                for value in values:
                    new_edge = row.copy()
                    new_edge['_end'] = value.strip()  # Clean up extra spaces
                    new_edge['_start'] = row[start_column]  # Use the _id as the _start for the edge
                    new_edge['_type'] = edge_type
                    expanded_edges.append(new_edge)
        return pd.DataFrame(expanded_edges)

    # Create edges for related organs
    organ_edges = split_and_create_edges(case_study_df, 'related_organ', 'case_study_related_organ')

    # Create edges for related AOPs
    case_study_aop_edges = split_and_create_edges(case_study_df, 'related_aop', 'case_study_related_aop')

    # Create edges for related chemicals
    case_study_chemical_edges = split_and_create_edges(case_study_df, 'related_chemical', 'case_study_relevant_chemical')

    # Create edges for related model systems
    model_system_edges = split_and_create_edges(case_study_df, 'related_model_system', 'case_study_relevant_model_system')

    # Create edges for related computational models
    computational_model_edges = split_and_create_edges(case_study_df, 'related_computational_model', 'case_study_relevant_computational_model')

    # Chemical-measured-with-bioassay
    chemical_bioassay_edges = split_and_create_edges(chemical_df, 'measured_with_bioassay', 'chemical_measured_with_bioassay')

    # Bioassay-executed-on-model_system
    bioassay_model_system_edges = split_and_create_edges(bioassay_df, 'related_model_system', 'bioassay_executed_on_model_system')

    # Bioassay-related-organ
    bioassay_organ_edges = split_and_create_edges(bioassay_df, 'related_organ', 'bioassay_related_organ')

    # Edge for chemical_relevant_to_computational_model
    chemical_computational_model_edges = split_and_create_edges(chemical_df, 'relevant_computational_model', 'chemical_relevant_to_computational_model')

    # Edge for chemical_measured_in_model_system
    chemical_model_system_edges = split_and_create_edges(chemical_df, 'measured_in_model_system', 'chemical_measured_in_model_system')

    # Edge for model_system_relevant_to_organ
    model_system_organ_edges = split_and_create_edges(model_system_df, 'relevant_organ', 'model_system_relevant_to_organ')

    # Edge for computational_model_relevant_to_organ
    computational_model_organ_edges = split_and_create_edges(computational_model_df, 'relevant_organ', 'computational_model_relevant_to_organ')

    # Edge for case_study_relevant_endpoint
    case_study_endpoint_edges = split_and_create_edges(case_study_df, 'related_endpoint', 'case_study_relevant_endpoint')

    # Edge for bioassay_used_with_experimental_condition
    bioassay_condition_edges = split_and_create_edges(bioassay_df, 'used_with_experimental_condition', 'bioassay_used_with_experimental_condition')

    # Combine all edges into one DataFrame
    edges = pd.concat([
        organ_edges, case_study_aop_edges, case_study_chemical_edges, model_system_edges, computational_model_edges, 
        chemical_bioassay_edges, bioassay_model_system_edges, bioassay_organ_edges, 
        chemical_computational_model_edges, chemical_model_system_edges, model_system_organ_edges, 
        computational_model_organ_edges, case_study_endpoint_edges, bioassay_condition_edges
    ], ignore_index=True)

    return edges



def save_combined_csv():
    case_study_df, chemical_df, bioassay_df, model_system_df, computational_model_df, final_nodes = read_csv_files()

    # Create edge table from relationships
    edges_df = create_edges(case_study_df, chemical_df, bioassay_df, model_system_df, computational_model_df)

    # Combine nodes and edges
    combined_df = pd.concat([final_nodes, edges_df], ignore_index=True)

    # Sort the combined data by _id and then by _start/_end to maintain node-then-edge order
    combined_df = combined_df.sort_values(by=['_id', '_start'], na_position='first')

    # Save the output to a new CSV file
    combined_df.to_csv("Combined_output.csv", index=False)

    print("Combined CSV saved as 'Combined_output.csv'.")

if __name__ == "__main__":
    save_combined_csv()
