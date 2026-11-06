# Integrated Street Canyon H/W Analyzer
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class IntegratedHWAnalyzer:
    """
    Integrated street canyon H/W analysis
    Directly reads from CSV file
    """

    def __init__(self):
        # H/W classification standards (based on literature)
        self.hw_categories = {
            'non_canyon': (-1, 0),
            'shallow': (0, 1),  # Shallow canyon
            'regular': (1, 2),  # Regular canyon
            'deep': (2, 4),  # Deep canyon
            'very_deep': (4, 100)  # Very deep canyon
        }

    def load_data(self, csv_file: str) -> pd.DataFrame:
        """
        Load semantic segmentation results from CSV file
        """
        if os.path.exists(csv_file):
            print(f"Reading file: {csv_file}")
            df = pd.read_csv(csv_file)

            # Ensure required columns exist
            required_columns = ['image', 'building', 'sky', 'road', 'sidewalk']
            for col in required_columns:
                if col not in df.columns:
                    raise ValueError(f"CSV file missing required column: {col}")

            return df
        else:
            raise FileNotFoundError(f"File {csv_file} not found")

    def extract_point_info(self, filename: str) -> tuple:
        """
        Extract point ID and angle from filename
        Format: 1_121.620895_38.918283_0.png
        """
        try:
            parts = filename.split('_')
            point_id = int(parts[0])
            angle = int(parts[-1].split('.')[0])
            return point_id, angle
        except Exception as e:
            print(f"Error parsing filename: {filename} - {e}")
            return -1, -1

    def calculate_hw_from_segmentation(self, row: pd.Series) -> dict:
        """
        Calculate H/W-related features based on semantic segmentation results
        """
        # Building-related classes
        building_ratio = row.get('building', 0) + row.get('wall', 0) + row.get('fence', 0)

        # Sky
        sky_ratio = row.get('sky', 0)

        # Road-related
        road_ratio = row.get('road', 0) + row.get('sidewalk', 0)

        # Vegetation
        vegetation_ratio = row.get('vegetation', 0) + row.get('terrain', 0)

        # Estimate H/W value
        if sky_ratio > 0.01:  # Avoid division by zero
            # Empirical formula based on building and sky ratios
            hw_estimate = (building_ratio / (sky_ratio + 0.1)) * 1.5

            # Adjust based on road width
            if road_ratio > 0.3:  # Wide road
                hw_estimate *= 0.7
            elif road_ratio < 0.1:  # Narrow road
                hw_estimate *= 1.3

            # Consider absolute building ratio
            if building_ratio > 0.6:  # Building-dominated
                hw_estimate *= 1.2
            elif building_ratio < 0.2:  # Few buildings
                hw_estimate *= 0.8
        else:
            # Sky not visible, consider as very deep canyon
            hw_estimate = 5.0

        # Limit range
        hw_estimate = np.clip(hw_estimate, 0.1, 10.0)

        # Determine category
        hw_category = 'non_canyon'
        for cat, (min_val, max_val) in self.hw_categories.items():
            if min_val < hw_estimate <= max_val:
                hw_category = cat
                break

        return {
            'hw_estimate': hw_estimate,
            'hw_category': hw_category,
            'building_ratio': building_ratio,
            'sky_ratio': sky_ratio,
            'road_ratio': road_ratio,
            'vegetation_ratio': vegetation_ratio
        }

    def analyze_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Analyze data and return results (only average values for each point)
        """
        # Extract point ID and angle
        df[['point_id', 'angle']] = df['image'].apply(
            lambda x: pd.Series(self.extract_point_info(x))
        )

        # Filter invalid points
        valid_df = df[df['point_id'] > 0]
        invalid_count = len(df) - len(valid_df)
        if invalid_count > 0:
            print(f"Warning: Found {invalid_count} invalid point IDs")

        # Analyze by point groups
        results = []

        # Group by point
        grouped = valid_df.groupby('point_id')

        for point_id, group in grouped:
            # Calculate H/W features for each image
            hw_results = []

            for _, row in group.iterrows():
                hw_result = self.calculate_hw_from_segmentation(row)
                hw_results.append(hw_result)

            # Calculate averages for this point
            avg_hw = np.mean([r['hw_estimate'] for r in hw_results])
            avg_building = np.mean([r['building_ratio'] for r in hw_results])
            avg_sky = np.mean([r['sky_ratio'] for r in hw_results])
            avg_road = np.mean([r['road_ratio'] for r in hw_results])
            avg_vegetation = np.mean([r['vegetation_ratio'] for r in hw_results])

            # Calculate standard deviation (for symmetry)
            hw_std = np.std([r['hw_estimate'] for r in hw_results])
            building_std = np.std([r['building_ratio'] for r in hw_results])

            # Determine final category
            final_category = 'non_canyon'
            for cat, (min_val, max_val) in self.hw_categories.items():
                if min_val < avg_hw <= max_val:
                    final_category = cat
                    break

            # Determine symmetry
            symmetry = 'symmetric' if hw_std < 0.5 else 'asymmetric'

            # Calculate confidence
            confidence = 1.0
            if hw_std > 0:
                confidence *= (1 / (1 + hw_std))  # Higher std, lower confidence
            confidence = np.clip(confidence, 0.3, 1.0)

            result = {
                'point_id': int(point_id),
                'num_images': len(group),
                'hw_value': round(avg_hw, 2),
                'hw_category': final_category,
                'hw_std': round(hw_std, 3),
                'symmetry': symmetry,
                'confidence': round(confidence, 3),
                'avg_building_ratio': round(avg_building, 3),
                'avg_sky_ratio': round(avg_sky, 3),
                'avg_road_ratio': round(avg_road, 3),
                'avg_vegetation_ratio': round(avg_vegetation, 3),
                'building_std': round(building_std, 3)
            }

            results.append(result)

        # Create results DataFrame and sort
        results_df = pd.DataFrame(results)
        if not results_df.empty:
            results_df = results_df.sort_values('point_id')

        return results_df

    def generate_report(self, results_df: pd.DataFrame, output_file: str):
        """
        Generate analysis report (only CSV output)
        """
        # Save CSV
        results_df.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"\nResults saved to: {output_file}")

        # Print summary statistics
        print("\n" + "=" * 60)
        print("Street Canyon H/W Analysis Results")
        print("=" * 60)

        if results_df.empty:
            print("Warning: No valid results generated")
            return

        print(f"\n1. Basic Statistics")
        print(f"   Points analyzed: {len(results_df)}")
        print(f"   Average H/W value: {results_df['hw_value'].mean():.2f}")
        print(f"   H/W std dev: {results_df['hw_value'].std():.2f}")
        print(f"   H/W range: {results_df['hw_value'].min():.2f} - {results_df['hw_value'].max():.2f}")

        print(f"\n2. H/W Category Distribution")
        category_counts = results_df['hw_category'].value_counts()
        for category in ['non_canyon', 'shallow', 'regular', 'deep', 'very_deep']:
            if category in category_counts.index:
                count = category_counts[category]
                percentage = count / len(results_df) * 100
                print(f"   {category:12s}: {count:3d} ({percentage:5.1f}%)")

        print(f"\n3. Symmetry Distribution")
        symmetry_counts = results_df['symmetry'].value_counts()
        for sym_type, count in symmetry_counts.items():
            percentage = count / len(results_df) * 100
            print(f"   {sym_type:12s}: {count:3d} ({percentage:5.1f}%)")

        print(f"\n4. Deep Canyons (H/W > 2)")
        deep_canyons = results_df[results_df['hw_value'] > 2].sort_values('hw_value', ascending=False)
        print(f"   Found {len(deep_canyons)} deep canyon points")
        if len(deep_canyons) > 0:
            print("   Top 10 deepest canyons:")
            for _, row in deep_canyons.head(10).iterrows():
                print(f"      Point {row['point_id']:3d}: H/W = {row['hw_value']:4.2f} ({row['hw_category']})")

        # Generate visualizations only
        self.generate_visualizations(results_df)

    def generate_visualizations(self, results_df: pd.DataFrame):
        """
        Generate analysis visualizations (in English)
        """
        try:
            # Create output directory for plots
            os.makedirs('visualizations', exist_ok=True)

            # H/W Value Distribution
            plt.figure(figsize=(12, 8))

            # H/W Value Histogram
            plt.subplot(2, 2, 1)
            plt.hist(results_df['hw_value'], bins=20, color='skyblue', edgecolor='black')
            plt.xlabel('H/W Value')
            plt.ylabel('Frequency')
            plt.title('H/W Value Distribution')
            plt.grid(True, linestyle='--', alpha=0.7)

            # Category Distribution
            plt.subplot(2, 2, 2)
            category_counts = results_df['hw_category'].value_counts()
            colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99', '#c2c2f0']
            plt.pie(category_counts.values, labels=category_counts.index, autopct='%1.1f%%',
                    colors=colors, startangle=90)
            plt.axis('equal')
            plt.title('H/W Category Distribution')

            # Building Ratio vs H/W
            plt.subplot(2, 2, 3)
            plt.scatter(results_df['avg_building_ratio'], results_df['hw_value'],
                        alpha=0.6, c='green', edgecolors='w')
            plt.xlabel('Average Building Ratio')
            plt.ylabel('H/W Value')
            plt.title('Building Ratio vs H/W Value')
            plt.grid(True, linestyle='--', alpha=0.7)

            # Sky Ratio vs H/W
            plt.subplot(2, 2, 4)
            plt.scatter(results_df['avg_sky_ratio'], results_df['hw_value'],
                        alpha=0.6, c='blue', edgecolors='w')
            plt.xlabel('Average Sky Ratio')
            plt.ylabel('H/W Value')
            plt.title('Sky Ratio vs H/W Value')
            plt.grid(True, linestyle='--', alpha=0.7)

            plt.tight_layout()
            plt.savefig('visualizations/hw_analysis_visualization.png', dpi=300)
            print("Visualization saved to: visualizations/hw_analysis_visualization.png")

            # Symmetry Analysis
            plt.figure(figsize=(10, 6))
            symmetry_counts = results_df['symmetry'].value_counts()
            plt.bar(symmetry_counts.index, symmetry_counts.values, color=['#1f77b4', '#ff7f0e'])
            plt.xlabel('Symmetry')
            plt.ylabel('Number of Points')
            plt.title('Street Canyon Symmetry Distribution')
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.savefig('visualizations/symmetry_distribution.png', dpi=300)
            print("Symmetry chart saved to: visualizations/symmetry_distribution.png")

            # Deep Canyon Analysis
            deep_canyons = results_df[results_df['hw_value'] > 2]
            if not deep_canyons.empty:
                plt.figure(figsize=(10, 6))
                plt.scatter(deep_canyons['point_id'], deep_canyons['hw_value'],
                            c=deep_canyons['confidence'], cmap='viridis', s=100, alpha=0.7)
                plt.colorbar(label='Confidence')
                plt.xlabel('Point ID')
                plt.ylabel('H/W Value')
                plt.title('Deep Canyon Points (H/W > 2)')
                plt.grid(True, linestyle='--', alpha=0.5)
                plt.savefig('visualizations/deep_canyons.png', dpi=300)
                print("Deep canyon plot saved to: visualizations/deep_canyons.png")

            plt.close('all')

        except Exception as e:
            print(f"Error generating visualizations: {e}")


def main(csv_file: str):
    """
    Main function: Load data and analyze H/W values
    """
    # Create analyzer
    analyzer = IntegratedHWAnalyzer()

    try:
        # Load data
        print("Loading semantic segmentation results...")
        seg_df = analyzer.load_data(csv_file)
        print(f"Successfully read {len(seg_df)} records")

        # Analyze data
        print("\nAnalyzing H/W values...")
        results_df = analyzer.analyze_data(seg_df)

        if results_df.empty:
            print("Warning: No valid results generated")
            return

        # Generate report (only CSV and visualizations)
        output_file = 'street_canyon_hw_analysis_results.csv'
        analyzer.generate_report(results_df, output_file)

        return results_df

    except Exception as e:
        print(f"Error during processing: {e}")
        return None


if __name__ == "__main__":
    # Specify your CSV file path here
    csv_file_path = "./all_image_results.csv"

    # Run analysis
    results = main(csv_file_path)

    if results is not None:
        print("\nAnalysis completed successfully!")
        print(f"Results saved to: street_canyon_hw_analysis_results.csv")
        print(f"Visualizations saved to: visualizations/ folder")