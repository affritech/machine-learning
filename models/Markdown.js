import mongoose from "mongoose";

// Define the schema for markdown documents
const markdownSchema = new mongoose.Schema({
  // Document title
  title: {
    type: String,
    required: [true, 'Title is required'],
    trim: true,  // Removes whitespace from beginning and end
    maxlength: [200, 'Title cannot exceed 200 characters']
  },
  
  // The markdown content
  content: {
    type: String,
    required: [true, 'Content is required']
  },
  
  // Optional description/summary
  description: {
    type: String,
    trim: true,
    maxlength: [500, 'Description cannot exceed 500 characters']  // Fixed: was 5000
  },
  
  // Tags as an array of strings
  tags: [{  // Array notation with []
    type: String,
    trim: true
  }],
  
  // Creation timestamp
  createdAt: {  // Fixed typo: was 'createdAT'
    type: Date,
    default: Date.now
  },
  
  // Last update timestamp
  updatedAt: {
    type: Date,
    default: Date.now
  }
});

// Middleware: Update 'updatedAt' before saving
markdownSchema.pre('save', function(next) {
  this.updatedAt = Date.now();
  next();
});

// Create and export the model
const Markdown = mongoose.model('Markdown', markdownSchema);

export default Markdown;