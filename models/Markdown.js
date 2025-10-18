import mongoose from "mongoose";


const markdownSchema = new mongoose.Schema({
   title: {
        type: String,
        required: [true, 'Title is required'],
        trim: true,
        maxlength: [200, 'Title cannot exceed 200 characters']
    },
    content: {
        type: String,
        required: [true, 'Content is required']
    },
    description: {
        type: String,
        trim: true,
        maxlength: [500, 'description cannot exceed 5000 characters']
    },
    tags: {
        type: String,
        trim: true,

    },
    createdAT: {
        type: Date,
        default: Date.now

    },
    updatedAt: {
        type: Date,
        default: Date.now
    }
}

);

// update before saving 

markdownSchema.pre('save', function(next){
    this.updatedAt = Date.now();
    next();
});

// create and export model

const Markdown = mongoose.model('Markdown', markdownSchema );

export default Markdown;