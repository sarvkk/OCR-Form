
### 2. Using the Pipeline on Real Forms

To use the trained pipeline on your own KYC forms:

```python
from kyc_form_detector import KYCFormTextDetection

# Initialize the pipeline
kyc_pipeline = KYCFormTextDetection(
    tesseract_path=r'C:\Program Files\Tesseract-OCR\tesseract.exe',  # Update with your path
    model_path='kyc_form_detector.h5'  # Path to your trained model
)

# Process a KYC form
result = kyc_pipeline.process_kyc_form('path/to/kyc_form.jpg')

# Display the extracted information
print("Extracted KYC Information:")
for field, value in result.items():
    print(f"{field}: {value}")

# Export results to JSON
kyc_pipeline.export_results(result, 'output_path/kyc_results.json')
```

3. Batch Processing Multiple Forms

```python
# Process all forms in a directory
form_directory = 'path/to/kyc_forms/'
output_directory = 'path/to/output/'

# Create the output directory if it doesn't exist
import os
os.makedirs(output_directory, exist_ok=True)

# Process all image files in the directory
import glob
form_files = glob.glob(os.path.join(form_directory, '*.jpg')) + \
             glob.glob(os.path.join(form_directory, '*.png')) + \
             glob.glob(os.path.join(form_directory, '*.pdf'))

for form_file in form_files:
    file_name = os.path.basename(form_file)
    print(f"Processing {file_name}...")
    
    try:
        # Process the form
        result = kyc_pipeline.process_kyc_form(form_file)
        
        # Export results to JSON
        output_file = os.path.join(output_directory, f"{os.path.splitext(file_name)[0]}.json")
        kyc_pipeline.export_results(result, output_file)
        
        print(f"Successfully processed {file_name}")
    except Exception as e:
        print(f"Error processing {file_name}: {str(e)}")
```

4. Integration with Existing Systems

The KYC pipeline can be integrated with your existing systems:

```python
# Example integration with a web API
def process_uploaded_form(uploaded_file):
    import tempfile
    
    # Save the uploaded file to a temporary location
    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
        temp_file_path = temp_file.name
        temp_file.write(uploaded_file.read())
    
    # Process the form
    try:
        kyc_pipeline = KYCFormTextDetection()
        result = kyc_pipeline.process_kyc_form(temp_file_path)
        
        # Clean up the temporary file
        os.remove(temp_file_path)
        
        return result
    except Exception as e:
        # Clean up the temporary file
        os.remove(temp_file_path)
        raise e
```

5. Fine-tuning the Model

To fine-tune the model on your specific KYC forms:

```python
from training_script import fine_tune_model

# Path to your annotated KYC forms
custom_forms_path = 'path/to/custom_forms/'
annotations_path = 'path/to/annotations.json'

# Fine-tune the model
fine_tune_model('kyc_form_detector.h5', custom_forms_path, annotations_path)
```

6. Performance Monitoring and Improvement

To monitor the performance of your KYC pipeline:

```python
# Evaluate the model on a test set
from training_script import evaluate_model

test_forms_path = 'path/to/test_forms/'
test_annotations_path = 'path/to/test_annotations.json'

evaluation_results = evaluate_model('kyc_form_detector.h5', test_forms_path, test_annotations_path)
print(f"Model accuracy: {evaluation_results['accuracy']:.2f}")
print(f"Field detection precision: {evaluation_results['precision']:.2f}")
print(f"Field detection recall: {evaluation_results['recall']:.2f}")
```

Troubleshooting
1. OCR Quality Issues
   - Ensure images are high resolution (at least 300 DPI)
   - Try preprocessing with different parameters
   - Consider using a custom OCR language pack for specific documents

2. Field Detection Problems
   - Increase the training data size
   - Add more variations to the synthetic form generator
   - Fine-tune the model with real KYC forms

3. Performance Issues
   - Reduce image resolution for faster processing
   - Implement batch processing for multiple forms
   - Use GPU acceleration if available

Best Practices
1. Data Security
   - Implement encryption for extracted KYC data
   - Delete temporary files after processing
   - Follow data protection regulations for storing personal information

2. Quality Assurance
   - Implement confidence scoring for extracted text
   - Flag low-confidence extractions for manual review
   - Periodically validate extraction accuracy with ground truth data

3. Continuous Improvement
   - Collect correction data from manual reviews
   - Regularly retrain the model with new examples
   - Track performance metrics over time

Now you have a complete KYC Form Text Detection Pipeline that can be used in production environments. This system handles form detection, text extraction, and post-processing to deliver accurate results from KYC documents.