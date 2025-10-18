import express from 'express';
const router = express.Router();
import { marked } from 'marked';
import Markdown from '../models/Markdown.js';

// Configure marked options (optional)
marked.setOptions({
  breaks: true, // Convert \n to <br>
  gfm: true,    // Use GitHub Flavored Markdown
});

/**
 * @route   GET /api/markdown
 * @desc    Get all markdown documents (converted to HTML)
 * @access  Public
 */
router.get('/', async (req, res) => {
  try {
    // Fetch all markdown documents from database
    const markdownDocs = await Markdown.find().sort({ createdAt: -1 });
    
    // Convert each document's markdown content to HTML
    const htmlDocs = markdownDocs.map(doc => ({
      id: doc._id,
      title: doc.title,
      description: doc.description,
      content: marked(doc.content), // Convert Markdown to HTML
      htmlContent: marked(doc.content), // Also provide as htmlContent
      rawMarkdown: doc.content, // Keep original markdown
      tags: doc.tags,
      createdAt: doc.createdAt,
      updatedAt: doc.updatedAt
    }));

    res.json({
      success: true,
      count: htmlDocs.length,
      data: htmlDocs
    });
  } catch (error) {
    console.error('Error fetching markdown documents:', error);
    res.status(500).json({
      success: false,
      message: 'Server error while fetching documents',
      error: error.message
    });
  }
});

/**
 * @route   GET /api/markdown/:id
 * @desc    Get single markdown document by ID (converted to HTML)
 * @access  Public
 */
router.get('/:id', async (req, res) => {
  try {
    const doc = await Markdown.findById(req.params.id);
    
    if (!doc) {
      return res.status(404).json({
        success: false,
        message: 'Document not found'
      });
    }

    // Convert markdown to HTML
    const htmlDoc = {
      id: doc._id,
      title: doc.title,
      description: doc.description,
      content: marked(doc.content), // HTML version
      htmlContent: marked(doc.content),
      rawMarkdown: doc.content, // Original markdown
      tags: doc.tags,
      createdAt: doc.createdAt,
      updatedAt: doc.updatedAt
    };

    res.json({
      success: true,
      data: htmlDoc
    });
  } catch (error) {
    console.error('Error fetching document:', error);
    
    // Handle invalid MongoDB ID format
    if (error.kind === 'ObjectId') {
      return res.status(404).json({
        success: false,
        message: 'Document not found'
      });
    }
    
    res.status(500).json({
      success: false,
      message: 'Server error while fetching document',
      error: error.message
    });
  }
});

/**
 * @route   POST /api/markdown
 * @desc    Create a new markdown document
 * @access  Public
 */
router.post('/', async (req, res) => {
  try {
    const { title, content, description, tags } = req.body;

    // Validation
    if (!title || !content) {
      return res.status(400).json({
        success: false,
        message: 'Title and content are required'
      });
    }

    // Create new document
    const newDoc = await Markdown.create({
      title,
      content,
      description,
      tags
    });

    // Return the created document with HTML conversion
    const htmlDoc = {
      id: newDoc._id,
      title: newDoc.title,
      description: newDoc.description,
      content: marked(newDoc.content),
      htmlContent: marked(newDoc.content),
      rawMarkdown: newDoc.content,
      tags: newDoc.tags,
      createdAt: newDoc.createdAt,
      updatedAt: newDoc.updatedAt
    };

    res.status(201).json({
      success: true,
      message: 'Document created successfully',
      data: htmlDoc
    });
  } catch (error) {
    console.error('Error creating document:', error);
    
    // Handle validation errors
    if (error.name === 'ValidationError') {
      const messages = Object.values(error.errors).map(err => err.message);
      return res.status(400).json({
        success: false,
        message: 'Validation error',
        errors: messages
      });
    }
    
    res.status(500).json({
      success: false,
      message: 'Server error while creating document',
      error: error.message
    });
  }
});

/**
 * @route   PUT /api/markdown/:id
 * @desc    Update a markdown document
 * @access  Public
 */
router.put('/:id', async (req, res) => {
  try {
    const { title, content, description, tags } = req.body;

    // Find and update the document
    const doc = await Markdown.findByIdAndUpdate(
      req.params.id,
      { title, content, description, tags, updatedAt: Date.now() },
      { new: true, runValidators: true } // Return updated doc and run validators
    );

    if (!doc) {
      return res.status(404).json({
        success: false,
        message: 'Document not found'
      });
    }

    // Return updated document with HTML
    const htmlDoc = {
      id: doc._id,
      title: doc.title,
      description: doc.description,
      content: marked(doc.content),
      htmlContent: marked(doc.content),
      rawMarkdown: doc.content,
      tags: doc.tags,
      createdAt: doc.createdAt,
      updatedAt: doc.updatedAt
    };

    res.json({
      success: true,
      message: 'Document updated successfully',
      data: htmlDoc
    });
  } catch (error) {
    console.error('Error updating document:', error);
    
    if (error.kind === 'ObjectId') {
      return res.status(404).json({
        success: false,
        message: 'Document not found'
      });
    }
    
    res.status(500).json({
      success: false,
      message: 'Server error while updating document',
      error: error.message
    });
  }
});

/**
 * @route   DELETE /api/markdown/:id
 * @desc    Delete a markdown document
 * @access  Public
 */
router.delete('/:id', async (req, res) => {
  try {
    const doc = await Markdown.findByIdAndDelete(req.params.id);

    if (!doc) {
      return res.status(404).json({
        success: false,
        message: 'Document not found'
      });
    }

    res.json({
      success: true,
      message: 'Document deleted successfully',
      data: {
        id: doc._id,
        title: doc.title
      }
    });
  } catch (error) {
    console.error('Error deleting document:', error);
    
    if (error.kind === 'ObjectId') {
      return res.status(404).json({
        success: false,
        message: 'Document not found'
      });
    }
    
    res.status(500).json({
      success: false,
      message: 'Server error while deleting document',
      error: error.message
    });
  }
});

export default router;