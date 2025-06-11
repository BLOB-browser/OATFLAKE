#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test script to understand what fields the universal data saver puts in CSV files.
This will create sample data and show exactly what gets written to the CSV.
"""

import os
import tempfile
import shutil
import csv
import json
from datetime import datetime
from scripts.analysis.data_saver import DataSaver

def analyze_universal_csv_fields():
    """Analyze what fields are actually written to CSV files by the universal data saver"""
    
    # Create a temporary directory for testing
    test_dir = tempfile.mkdtemp()
    print(f"Using test directory: {test_dir}")
    
    try:
        # Initialize the DataSaver with test directory
        data_saver = DataSaver(test_dir)
        
        # Create comprehensive sample data covering all field scenarios
        test_items = [
            {
                'title': 'Sample Reference',
                'description': 'A comprehensive reference document for testing',
                'tags': ['reference', 'testing', 'documentation'],
                'purpose': 'To test the universal data saver functionality',
                'location': 'https://example.com/reference',
                'origin_url': 'https://example.com/source',
                'related_url': 'https://example.com/related',
                'status': 'active',
                'creator_id': 'test_user',
                'collaborators': ['user1', 'user2'],
                'group_id': 'test_group',
                'visibility': 'public',
                'analysis_completed': True,
                'custom_field': 'This should be preserved'
            },
            {
                # Minimal item to test defaults
                'title': 'Minimal Reference',
                'description': 'Testing minimal field requirements'
            }
        ]
        
        print("=" * 80)
        print("TESTING UNIVERSAL DATA SAVER CSV OUTPUT")
        print("=" * 80)
        
        # Test with 'reference' content type
        print(f"\n📝 Testing with content_type: 'reference'")
        data_saver.save_universal_content(test_items, 'reference')
        
        # Read and analyze the resulting CSV
        csv_path = os.path.join(test_dir, 'references.csv')
        
        if os.path.exists(csv_path):
            print(f"✅ CSV created at: {csv_path}")
            
            # Read CSV and analyze structure
            with open(csv_path, 'r', newline='', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                
                # Get field names (column headers)
                fieldnames = reader.fieldnames
                print(f"\n📋 CSV COLUMNS ({len(fieldnames)} total):")
                print("-" * 60)
                
                for i, field in enumerate(fieldnames, 1):
                    print(f"{i:2d}. {field}")
                
                # Read and analyze data rows
                rows = list(reader)
                print(f"\n📊 CSV DATA ({len(rows)} rows):")
                print("-" * 60)
                
                for row_idx, row in enumerate(rows, 1):
                    print(f"\nRow {row_idx}:")
                    for field, value in row.items():
                        # Truncate long values for display
                        display_value = value
                        if len(str(value)) > 50:
                            display_value = str(value)[:47] + "..."
                        print(f"  {field}: {repr(display_value)}")
                
                # Analyze field types and patterns
                print(f"\n🔍 FIELD ANALYSIS:")
                print("-" * 60)
                
                if rows:
                    sample_row = rows[0]
                    for field, value in sample_row.items():
                        # Determine field characteristics
                        field_type = "string"
                        is_json = False
                        is_timestamp = False
                        is_boolean = False
                        
                        try:
                            # Check if it's JSON
                            json.loads(value)
                            is_json = True
                            field_type = "JSON"
                        except:
                            pass
                        
                        # Check if it's a timestamp
                        if 'at' in field.lower() and ('T' in value or '-' in value):
                            is_timestamp = True
                            field_type = "timestamp"
                        
                        # Check if it's boolean
                        if value.lower() in ['true', 'false']:
                            is_boolean = True
                            field_type = "boolean"
                        
                        print(f"  {field:20} | {field_type:10} | {repr(value[:30])}")
                
                # Test different content types for comparison
                print(f"\n🔄 TESTING OTHER CONTENT TYPES:")
                print("-" * 60)
                
                content_types_to_test = ['method', 'definition', 'project']
                
                for content_type in content_types_to_test:
                    # Create type-specific test data
                    type_specific_items = [{
                        'title': f'Sample {content_type.title()}',
                        'description': f'A test {content_type} for field analysis',
                        'tags': [content_type, 'test']
                    }]
                    
                    # Add content-type specific fields
                    if content_type == 'method':
                        type_specific_items[0]['steps'] = ['Step 1', 'Step 2', 'Step 3']
                    elif content_type == 'definition':
                        type_specific_items[0]['term'] = f'Test {content_type.title()}'
                        type_specific_items[0]['definition'] = f'Definition of {content_type}'
                    elif content_type == 'project':
                        type_specific_items[0]['goals'] = ['Goal 1', 'Goal 2']
                        type_specific_items[0]['achievement'] = 'Completed successfully'
                    
                    data_saver.save_universal_content(type_specific_items, content_type)
                    
                    type_csv_path = os.path.join(test_dir, f'{content_type}s.csv')
                    if os.path.exists(type_csv_path):
                        with open(type_csv_path, 'r', newline='', encoding='utf-8') as f:
                            type_reader = csv.DictReader(f)
                            type_fieldnames = type_reader.fieldnames
                            print(f"\n  {content_type.upper()} CSV columns ({len(type_fieldnames)}):")
                            print(f"    {', '.join(type_fieldnames)}")
                
        else:
            print("❌ CSV was not created!")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Clean up test directory
        if os.path.exists(test_dir):
            shutil.rmtree(test_dir)
            print(f"\n🧹 Cleaned up test directory: {test_dir}")

if __name__ == "__main__":
    print("=" * 80)
    print("UNIVERSAL DATA SAVER CSV FIELD ANALYSIS")
    print("=" * 80)
    
    success = analyze_universal_csv_fields()
    
    print("\n" + "=" * 80)
    if success:
        print("🎉 Analysis completed successfully!")
    else:
        print("💥 Analysis failed!")
    print("=" * 80)
