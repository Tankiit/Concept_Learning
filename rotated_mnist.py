import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from transformers import AutoTokenizer, AutoModel
import torchvision.transforms.functional as TF
from config import get_config

import PIL
from PIL import Image
import os
import numpy as np
import gc
config = get_config()


class RotatedMNISTv3(Dataset):
    def __init__(self, config, train=True, transform=None):
        super().__init__()
        self.config = config
        self.root_dir = self.config.data.root_dir
        self.train = train
        self.angles = self.config.data.angles
        
        # Disable tokenizer parallelism
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        
        # Define default transform if none provided
        if transform is None:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((config.data.mean,), (config.data.std,)),
                transforms.RandomAffine(
                    degrees=0, 
                    translate=(0.1, 0.1), 
                    scale=(0.9, 1.1)
                ) if train else transforms.Lambda(lambda x: x)
            ])
        else:
            self.transform = transform
        
        # Load MNIST dataset
        self.mnist = datasets.MNIST(
            root=self.root_dir,
            train=train,
            download=True,
            transform=None
        )
        
        # Initialize shared concept cache using shared memory
        self.concept_cache = {}
        
        # Lazy load concept extractor and precompute in batches
        self._concept_extractor = None
        self._precompute_done = False
        
        print(f"Initialized RotatedMNIST dataset:")
        print(f"- Split: {'Train' if train else 'Test'}")
        print(f"- Samples: {len(self)}")
        print(f"- Angles: {self.angles}")
        
        # Precompute concepts in background if enabled
        if config.data.cache_concepts:
            self._precompute_concepts_async()
    
    @property
    def concept_extractor(self):
        """Lazy loading of concept extractor with memory optimization"""
        if self._concept_extractor is None:
            # Clear any unused memory before loading BERT
            gc.collect()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            self._concept_extractor = SimpleLLMConceptExtractorv1(self.config)
        return self._concept_extractor
    
    def _precompute_concepts_async(self):
        """Precompute concepts asynchronously in batches"""
        from concurrent.futures import ThreadPoolExecutor
        import threading
        
        def compute_batch(batch):
            results = {}
            for digit, angle in batch:
                if (digit, angle) not in self.concept_cache:
                    concepts = self.concept_extractor(digit, angle)
                    results[(digit, angle)] = {
                        'invariant_concepts': concepts['invariant_concepts'].cpu(),
                        'variant_concepts': concepts['variant_concepts'].cpu()
                    }
            return results
        
        def precompute_worker():
            # Generate all digit-angle combinations
            combinations = [(d, a) for d in range(10) for a in self.angles]
            
            # Split into batches
            batch_size = self.config.data.concept_batch_size
            batches = [combinations[i:i + batch_size] 
                      for i in range(0, len(combinations), batch_size)]
            
            # Process batches
            with ThreadPoolExecutor(max_workers=1) as executor:
                futures = []
                for batch in batches:
                    future = executor.submit(compute_batch, batch)
                    futures.append(future)
                
                # Collect results
                for future in futures:
                    results = future.result()
                    self.concept_cache.update(results)
            
            self._precompute_done = True
        
        # Start precomputation in background
        threading.Thread(target=precompute_worker, daemon=True).start()
    
    def _get_cached_concepts(self, digit, angle):
        """Get concepts with fallback for async precomputation"""
        cache_key = (int(digit), int(angle))
        
        # Try to get from cache
        if cache_key in self.concept_cache:
            return self.concept_cache[cache_key]
        
        # If not in cache, compute on demand
        concepts = self.concept_extractor(digit, angle)
        result = {
            'invariant_concepts': concepts['invariant_concepts'].cpu(),
            'variant_concepts': concepts['variant_concepts'].cpu()
        }
        
        # Cache only if precomputation isn't running
        if not self._precompute_done:
            self.concept_cache[cache_key] = result
            
        return result
    
    def __len__(self):
        return len(self.mnist) * len(self.angles)
    
    def __getitem__(self, idx):
        # Get base sample and rotation
        mnist_idx = idx // len(self.angles)
        angle_idx = idx % len(self.angles)
        angle = self.angles[angle_idx]
        
        # Get original image and digit
        image, digit = self.mnist[mnist_idx]
        
        try:
            # Apply transforms and rotation
            if isinstance(image, Image.Image):
                image = self.transform(image)
            rotated_image = TF.rotate(image, angle)
            
            # Get concepts
            concepts = self._get_cached_concepts(digit, angle)
            
            return {
                'image': rotated_image,
                'digit': torch.tensor(digit, dtype=torch.long),
                'angle': torch.tensor(angle_idx, dtype=torch.long),
                'invariant_concepts': concepts['invariant_concepts'],
                'variant_concepts': concepts['variant_concepts']
            }
            
        except Exception as e:
            print(f"Error in __getitem__ for idx {idx}: {str(e)}")
            # Return fallback sample
            return {
                'image': image,
                'digit': torch.tensor(digit, dtype=torch.long),
                'angle': torch.tensor(angle_idx, dtype=torch.long),
                'invariant_concepts': torch.zeros(self.concept_extractor.hidden_size),
                'variant_concepts': torch.zeros(self.concept_extractor.hidden_size)
            }



class RotatedMNISTv2(Dataset):
    def __init__(self, config, train=True, transform=None):
        super().__init__()
        self.config = config
        self.root_dir = config.data.root_dir
        self.train = train
        self.angles = self.config.data.angles
        
        # Set tokenizer parallelism explicitly
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        
        # Define default transform if none provided
        if transform is None:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((config.data.mean,), (config.data.std,)),
                transforms.RandomAffine(
                    degrees=0, 
                    translate=(0.1, 0.1), 
                    scale=(0.9, 1.1)
                ) if train else transforms.Lambda(lambda x: x)
            ])
        else:
            self.transform = transform
        
        # Load MNIST dataset
        self.mnist = datasets.MNIST(
            root=self.root_dir,
            train=train,
            download=True,
            transform=None
        )
        
        # Initialize shared concept cache
        self.concept_cache = {}
        
        # Initialize concept extractor (lazy loading)
        self._concept_extractor = None
        
        print(f"Initialized RotatedMNIST dataset:")
        print(f"- Split: {'Train' if train else 'Test'}")
        print(f"- Samples: {len(self)}")
        print(f"- Angles: {self.angles}")
        
        # Pre-compute concepts for all digit-angle pairs
        if config.data.cache_concepts:
            self._precompute_concepts()
    
    @property
    def concept_extractor(self):
        """Lazy loading of concept extractor"""
        if self._concept_extractor is None:
            self._concept_extractor = SimpleLLMConceptExtractorv1(self.config)
        return self._concept_extractor
    
    def _precompute_concepts(self):
        """Pre-compute concepts for all digit-angle pairs"""
        print("Pre-computing concepts...")
        for digit in range(10):
            for angle in self.angles:
                self._get_cached_concepts(digit, angle)
    
    def _get_cached_concepts(self, digit, angle):
        """Get concepts from cache or compute them"""
        cache_key = (int(digit), int(angle))
        if cache_key not in self.concept_cache:
            concepts = self.concept_extractor(digit, angle)
            self.concept_cache[cache_key] = {
                'invariant_concepts': concepts['invariant_concepts'].cpu(),
                'variant_concepts': concepts['variant_concepts'].cpu()
            }
        return self.concept_cache[cache_key]
    
    def __len__(self):
        return len(self.mnist) * len(self.angles)
    
    def __getitem__(self, idx):
        # Get base sample and rotation
        mnist_idx = idx // len(self.angles)
        angle_idx = idx % len(self.angles)
        angle = self.angles[angle_idx]
        
        # Get original image and digit
        image, digit = self.mnist[mnist_idx]
        
        try:
            # Apply transforms and rotation
            if isinstance(image, Image.Image):
                image = self.transform(image)
            rotated_image = TF.rotate(image, angle)
            
            # Get concepts from cache
            concepts = self._get_cached_concepts(digit, angle)
            
            return {
                'image': rotated_image,
                'digit': torch.tensor(digit, dtype=torch.long),
                'angle': torch.tensor(angle_idx, dtype=torch.long),
                'invariant_concepts': concepts['invariant_concepts'],
                'variant_concepts': concepts['variant_concepts']
            }
            
        except Exception as e:
            print(f"Error in __getitem__ for idx {idx}: {str(e)}")
            # Return a valid sample with zero concepts
            return {
                'image': image,
                'digit': torch.tensor(digit, dtype=torch.long),
                'angle': torch.tensor(angle_idx, dtype=torch.long),
                'invariant_concepts': torch.zeros(self.concept_extractor.hidden_size),
                'variant_concepts': torch.zeros(self.concept_extractor.hidden_size)
            }

def worker_init_fn(worker_id):
    """Initialize worker with proper settings"""
    # Set different seed for each worker
    np.random.seed(torch.initial_seed() % 2**32 + worker_id)
    # Disable tokenizer parallelism in workers
    os.environ["TOKENIZERS_PARALLELISM"] = "false"




class RotatedMNISTv1(Dataset):
    def __init__(self, config, root_dir, train=True, transform=None):
        super().__init__()
        self.config = config
        self.root_dir = config.data.root_dir
        self.train = train
        self.angles = self.config.data.angles
        self.device = self.config.training.device
        
        # Define default transform if none provided
        if transform is None:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((config.data.mean,), (config.data.std,)),
                transforms.RandomAffine(
                    degrees=0, 
                    translate=(0.1, 0.1), 
                    scale=(0.9, 1.1)
                ) if train else transforms.Lambda(lambda x: x)
            ])
        else:
            self.transform = transform
        
        # Load MNIST dataset
        self.mnist = datasets.MNIST(
            root=self.root_dir,
            train=train,
            download=True,
            transform=None  # We'll apply transforms later
        )
        
        # Initialize concept extractor
        self.concept_extractor = SimpleLLMConceptExtractorv1(config)
        self.concept_cache = {}
        
        print(f"Initialized RotatedMNIST dataset:")
        print(f"- Split: {'Train' if train else 'Test'}")
        print(f"- Samples: {len(self)}")
        print(f"- Angles: {self.angles}")
        print(f"- Device: {self.device}")

    def __len__(self):
        return len(self.mnist) * len(self.angles)
    
    def _get_base_sample(self, idx):
        """Get base MNIST sample"""
        base_idx = idx // len(self.angles)
        angle_idx = idx % len(self.angles)
        image, digit = self.mnist[base_idx]
        angle = self.angles[angle_idx]
        
        return image, digit, angle, base_idx, angle_idx
    
    def _apply_rotation(self, image, angle):
        """Apply rotation to image"""
        if not torch.is_tensor(image):
            image = self.transform(image)
        return TF.rotate(image, angle)
    
    def _get_cached_concepts(self, digit, angle):
        """Get concepts from cache or compute them"""
        cache_key = (int(digit), int(angle))
        if cache_key in self.concept_cache:
            return self.concept_cache[cache_key]
        
        concepts = self._extract_concepts(digit, angle)
        self.concept_cache[cache_key] = concepts
        return concepts
    
    def _extract_concepts(self, digit, angle):
        """Extract concepts for a digit-angle pair"""
        try:
            # Get digit-specific concepts (invariant)
            digit_concepts = {
                "primary": self.concept_extractor.digit_concepts[digit]["primary"],
                "structural": self.concept_extractor.digit_concepts[digit]["structural"],
                "components": self.concept_extractor.digit_concepts[digit]["components"],
                "topology": self.concept_extractor.digit_concepts[digit]["topology"]
            }
            
            # Get angle-specific concepts (variant)
            angle_concepts = {
                "base_desc": self.concept_extractor.angle_concepts[angle]["base_desc"],
                "angle_desc": self.concept_extractor.angle_concepts[angle]["angle_desc"],
                "visual_changes": self.concept_extractor.angle_concepts[angle]["visual_changes"],
                "spatial_relations": self.concept_extractor.angle_concepts[angle]["spatial_relations"]
            }
            
            # Create prompts
            invariant_prompt = self.concept_extractor.invariant_template.format(
                digit=digit,
                **digit_concepts
            )
            
            variant_prompt = self.concept_extractor.variant_template.format(
                digit=digit,
                angle=angle,
                **angle_concepts
            )
            
            # Get embeddings
            with torch.no_grad():
                invariant_concepts = self.concept_extractor.encode_prompt(invariant_prompt)
                variant_concepts = self.concept_extractor.encode_prompt(variant_prompt)
            
            return {
                'invariant_concepts': invariant_concepts.cpu().detach(),
                'variant_concepts': variant_concepts.cpu().detach()
            }
            
        except Exception as e:
            print(f"Error extracting concepts for digit {digit}, angle {angle}: {str(e)}")
            # Return zero tensors on error
            zero_tensor = torch.zeros(self.concept_extractor.model.config.hidden_size)
            return {
                'invariant_concepts': zero_tensor,
                'variant_concepts': zero_tensor
            }
    
    def __getitem__(self, idx):
        # Get base sample and rotation
        image, digit, angle, base_idx, angle_idx = self._get_base_sample(idx)
        
        try:
            # Apply transforms and rotation
            if isinstance(image, Image.Image):
                image = self.transform(image)
            rotated_image = self._apply_rotation(image, angle)
            
            # Get concepts
            concepts = self._get_cached_concepts(digit, angle)
            
            return {
                'image': rotated_image,
                'digit': torch.tensor(digit, dtype=torch.long),
                'angle': torch.tensor(angle_idx, dtype=torch.long),  # Use angle index
                'invariant_concepts': concepts['invariant_concepts'],
                'variant_concepts': concepts['variant_concepts']
            }
            
        except Exception as e:
            print(f"Error in __getitem__ for idx {idx}: {str(e)}")
            # Return a valid sample with zero concepts on error
            return {
                'image': image,
                'digit': torch.tensor(digit, dtype=torch.long),
                'angle': torch.tensor(angle_idx, dtype=torch.long),
                'invariant_concepts': torch.zeros(self.concept_extractor.model.config.hidden_size),
                'variant_concepts': torch.zeros(self.concept_extractor.model.config.hidden_size)
            }

class SimpleLLMConceptExtractorv1:
    def __init__(self, config, model_name="bert-base-uncased"):
        self.config = config
        self.model_name = model_name
        
        # Initialize tokenizer and model on CPU
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
        
        # Store model's hidden size
        self.hidden_size = self.model.config.hidden_size
        
        # Create concept cache
        self.concept_cache = {}
        
        # Templates for concept extraction
        self.invariant_template = """
        Analyze the digit {digit} characteristics:
        Primary shape: {primary}
        Structural details: {structural}
        Component parts: {components}
        Topological features: {topology}
        """
        
        self.variant_template = """
        Describe the digit {digit} when rotated {angle} degrees:
        Base orientation: {base_desc}
        After rotation: {angle_desc}
        Visual changes: {visual_changes}
        Spatial relations: {spatial_relations}
        """
        
        # Initialize concept dictionaries
        self._init_concept_dictionaries()
    
    def __call__(self, digit, angle):
        """Extract concepts for a digit-angle pair with comprehensive error handling"""
        try:
            # Input validation
            if not isinstance(digit, (int, np.integer)) or digit not in self.digit_concepts:
                raise ValueError(f"Invalid digit: {digit}")
            if not isinstance(angle, (int, np.integer)) or angle not in self.angle_concepts:
                raise ValueError(f"Invalid angle: {angle}")
            
            # Check cache
            cache_key = (int(digit), int(angle))
            if cache_key in self.concept_cache:
                return self.concept_cache[cache_key]
            
            # Get concept descriptions
            digit_info = self.digit_concepts[digit]
            angle_info = self.angle_concepts[angle]
            
            # Format prompts
            invariant_prompt = self.invariant_template.format(
                digit=digit,
                **digit_info
            )
            
            variant_prompt = self.variant_template.format(
                digit=digit,
                angle=angle,
                **angle_info
            )
            
            # Get embeddings
            invariant_concepts = self.encode_prompt(invariant_prompt)
            variant_concepts = self.encode_prompt(variant_prompt)
            
            # Cache and return results
            result = {
                'invariant_concepts': invariant_concepts,
                'variant_concepts': variant_concepts
            }
            self.concept_cache[cache_key] = result
            return result
            
        except Exception as e:
            print(f"Error extracting concepts for digit {digit}, angle {angle}: {str(e)}")
            zero_tensor = torch.zeros(self.hidden_size)
            return {
                'invariant_concepts': zero_tensor,
                'variant_concepts': zero_tensor
            }
    
    def encode_prompt(self, prompt):
        """Encode a prompt efficiently"""
        try:
            # Tokenize with proper padding and truncation
            tokens = self.tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            )
            
            # Get embeddings
            with torch.no_grad():
                outputs = self.model(**tokens)
                embeddings = outputs.last_hidden_state[:, 0]  # Use CLS token
            
            return embeddings.squeeze(0)
            
        except Exception as e:
            print(f"Error encoding prompt: {str(e)}")
            return torch.zeros(self.hidden_size)
    
    def _init_concept_dictionaries(self):
        """Initialize concept dictionaries for digits and angles"""
        # Pre-defined digit concepts (similar to your original implementation)
        self.digit_concepts = {
            0: {"primary": "closed loop", "structural": ["closed curve", "single continuous stroke", "hollow center"], "components": ["curved top", "curved bottom", "parallel sides"], "topology": "one enclosed region with no endpoints"},  # Add all digit concepts
            1: {"primary": "vertical line", "structural": ["single straight stroke", "minimal width", "vertical alignment"], "components": ["vertical line", "optional top flag", "sometimes small base"], "topology": "one or two endpoints with no intersections"},
            2: {"primary": "curved top connected to diagonal base stroke", "structural": ["upper clockwise curve", "diagonal sweep", "horizontal base"], "components": ["curved hook at top", "diagonal middle section", "flat bottom stroke"], "topology": "two endpoints with no intersections"},
            3: {"primary": "two connected curves forming right-facing semicircles", "structural": ["upper curve right", "lower curve right", "middle connection"], "components": ["top loop", "middle junction", "bottom loop"], "topology": "two endpoints with no intersections"},
            4: {"primary": "vertical and horizontal lines forming enclosed space", "structural": ["vertical stem", "horizontal crossbar", "diagonal or vertical connector"], "components": ["left vertical line", "horizontal crossing line", "right vertical descent"], "topology": "three endpoints with one intersection"},
            5: {"primary": "horizontal top with lower curved hook", "structural": ["top horizontal bar", "vertical descent", "bottom curve"], "components": ["flat top", "middle bend", "curved base"], "topology": "two endpoints with one self-intersection"},
            6: {"primary": "curved hook descending into bottom loop", "structural": ["upper curved stem", "counter-clockwise loop", "enclosed bottom"], "components": ["top curve or hook", "vertical descent", "bottom circular loop"], "topology": "one endpoint with one self-intersection"},
            7: {"primary": "horizontal top with diagonal descent", "structural": ["top horizontal bar", "diagonal stroke", "sometimes middle horizontal"], "components": ["flat top", "angled descent", "optional cross stroke"], "topology": "two endpoints with optional intersection"},
            8: {"primary": "two stacked loops forming closed regions", "structural": ["upper loop", "lower loop", "center crossing"], "components": ["top circle", "middle intersection", "bottom circle"], "topology": "no endpoints with central self-intersection"},
            9: {"primary": "top loop with curved descender", "structural": ["clockwise top loop", "vertical or curved tail", "right-side weighted"], "components": ["circular top", "right-side connection", "descending stroke"], "topology": "one endpoint with one self-intersection"}
        }
        
        # Pre-defined angle concepts (similar to your original implementation)
        self.angle_concepts = {
            0: {"base_desc": "standard upright", "angle_desc": "digit appears in its natural, upright position", "visual_changes": "no rotation applied, maintaining original appearance", "spatial_relations": "top remains at top, bottom at bottom, preserving vertical alignment", "reference_points": "all features maintain their expected positions relative to viewing direction"},  # Add all angle concepts
            90: {"base_desc": "rotated right with horizontal alignment", "angle_desc": "digit is turned 90 degrees clockwise, lying on its side", "visual_changes": "vertical elements become horizontal, horizontal elements become vertical", "spatial_relations": "original top points right, original bottom points left", "reference_points": "leftmost point becomes highest point, rightmost becomes lowest"},
            180: {"base_desc": "inverted upside down", "angle_desc": "digit is completely flipped, appearing upside down", "visual_changes": "complete inversion of all elements, mirror image across horizontal axis", "spatial_relations": "top becomes bottom, bottom becomes top, left and right swap", "reference_points": "highest points become lowest, and vice versa"},
            270: {"base_desc": "rotated left with horizontal alignment", "angle_desc": "digit is turned 90 degrees counterclockwise, lying on its side", "visual_changes": "vertical elements become horizontal, horizontal elements become vertical", "spatial_relations": "original top points left, original bottom points right", "reference_points": "rightmost point becomes highest point, leftmost becomes lowest"}
        }





class RotatedMNIST(Dataset):
    def __init__(self, config,root_dir, train=True, transform=None, angles=[0, 90, 180, 270], concept_type=None):
        super().__init__()
        self.config = config
        self.root_dir = root_dir
        self.train = train
        self.angles = angles
        
        # Define default transform if none provided
        if transform is None:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
        else:
            self.transform = transform
        
        # Load MNIST dataset
        self.mnist = datasets.MNIST(
            root=root_dir,
            train=train,
            download=True,
            transform=None  # Important: No transform here
        )
        
        # Setup concept extraction if enabled
        self.concept_type = concept_type
        if concept_type:
            self.concept_extractor = SimpleLLMConceptExtractor(config)
            self.concept_cache = {}
        
        print(f"Loaded RotatedMNIST: {len(self)} samples")
        print(f"Angles: {angles}")
        print(f"Concept extraction: {'Enabled' if concept_type else 'Disabled'}")
    
    def __len__(self):
        return len(self.mnist) * len(self.angles)
    
    def _get_concepts(self, digit, angle):
        """Get or compute concepts for a digit-angle pair
        
        Args:
            digit (int): The MNIST digit (0-9)
            angle (int): The rotation angle (0, 90, 180, 270)
            
        Returns:
            dict: Dictionary containing 'invariant_concepts' and 'variant_concepts' tensors
        """
        # Create cache key
        cache_key = (int(digit), int(angle))
        
        # Return cached concepts if available
        if cache_key in self.concept_cache:
            return self.concept_cache[cache_key]
        
        try:
            # Get concepts based on concept type
            if self.concept_type == 'simple':
                concepts = self.concept_extractor(digit, angle)
            elif self.concept_type == 'advanced':
                concepts = AdvancedLLMConceptExtractor()(digit, angle)
            else:
                raise ValueError(f"Unknown concept type: {self.concept_type}")
            
            # Cache the concepts
            self.concept_cache[cache_key] = {
                'invariant_concepts': concepts['invariant_concepts'].float(),
                'variant_concepts': concepts['variant_concepts'].float()
            }
            
            return self.concept_cache[cache_key]
            
        except Exception as e:
            print(f"Error extracting concepts for digit {digit}, angle {angle}: {e}")
            # Return zero tensors as fallback
            return {
                'invariant_concepts': torch.zeros(self.concept_extractor.model.config.hidden_size),
                'variant_concepts': torch.zeros(self.concept_extractor.model.config.hidden_size)
            }
    
    def __getitem__(self, idx):
        # Get original MNIST index and rotation angle
        mnist_idx = idx // len(self.angles)
        angle_idx = idx % len(self.angles)
        angle = self.angles[angle_idx]
        
        # Get original image and label
        image, digit = self.mnist[mnist_idx]
        
        # Ensure image is a tensor
        if isinstance(image, Image.Image):
            image = self.transform(image)  # Convert PIL Image to tensor
        
        # Rotate image (now working with tensor)
        if angle != 0:
            image = TF.rotate(image, angle)
        
        # Prepare output dictionary
        sample = {
            'image': image,  # Now guaranteed to be a tensor
            'digit': int(digit),  # Ensure digit is an integer
            'angle': angle_idx  # Use angle index
        }
        
        # Add concepts if enabled
        if self.concept_type:
            concepts = self._get_concepts(digit, angle)
            sample.update({
                'invariant_concepts': concepts['invariant_concepts'],
                'variant_concepts': concepts['variant_concepts']
            })
        
        return sample


# Example concept extractors
class SimpleLLMConceptExtractor:
    """Simple concept extractor using pre-defined LLM responses"""
    def __init__(self, config,model_name="bert-base-uncased"):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(self.config.training.device)
        self.model.eval()
         # Prompt templates
        self.invariant_template = """
        Describe the digit {digit}:
        Primary characteristics: {primary}
        Structural elements: {structural}
        Component parts: {components}
        Topological features: {topology}
        """
        
        self.variant_template = """
        Describe how digit {digit} appears when rotated {angle} degrees:
        Base orientation: {base_desc}
        Rotation effect: {angle_desc}
        """
        
        # Pre-defined concept descriptions
        self.digit_concepts = {
            0: {
                "primary": "continuous closed loop forming oval or circle",
                "structural": ["closed curve", "single continuous stroke", "hollow center"],
                "components": ["curved top", "curved bottom", "parallel sides"],
                "topology": "one enclosed region with no endpoints"
            },
            
            1: {
                "primary": "vertical straight line with optional serif",
                "structural": ["single straight stroke", "minimal width", "vertical alignment"],
                "components": ["vertical line", "optional top flag", "sometimes small base"],
                "topology": "one or two endpoints with no intersections"
            },
            
            2: {
                "primary": "curved top connected to diagonal base stroke",
                "structural": ["upper clockwise curve", "diagonal sweep", "horizontal base"],
                "components": ["curved hook at top", "diagonal middle section", "flat bottom stroke"],
                "topology": "two endpoints with no intersections"
            },
            
            3: {
                "primary": "two connected curves forming right-facing semicircles",
                "structural": ["upper curve right", "lower curve right", "middle connection"],
                "components": ["top loop", "middle junction", "bottom loop"],
                "topology": "two endpoints with one self-intersection"
            },
            
            4: {
                "primary": "vertical and horizontal lines forming enclosed space",
                "structural": ["vertical stem", "horizontal crossbar", "diagonal or vertical connector"],
                "components": ["left vertical line", "horizontal crossing line", "right vertical descent"],
                "topology": "three endpoints with one intersection"
            },
            
            5: {
                "primary": "horizontal top with lower curved hook",
                "structural": ["top horizontal bar", "vertical descent", "bottom curve"],
                "components": ["flat top", "middle bend", "curved base"],
                "topology": "two endpoints with one self-intersection"
            },
            
            6: {
                "primary": "curved hook descending into bottom loop",
                "structural": ["upper curved stem", "counter-clockwise loop", "enclosed bottom"],
                "components": ["top curve or hook", "vertical descent", "bottom circular loop"],
                "topology": "one endpoint with one self-intersection"
            },
            
            7: {
                "primary": "horizontal top with diagonal descent",
                "structural": ["top horizontal bar", "diagonal stroke", "sometimes middle horizontal"],
                "components": ["flat top", "angled descent", "optional cross stroke"],
                "topology": "two endpoints with optional intersection"
            },
            
            8: {
                "primary": "two stacked loops forming closed regions",
                "structural": ["upper loop", "lower loop", "center crossing"],
                "components": ["top circle", "middle intersection", "bottom circle"],
                "topology": "no endpoints with central self-intersection"
            },
            
            9: {
                "primary": "top loop with curved descender",
                "structural": ["clockwise top loop", "vertical or curved tail", "right-side weighted"],
                "components": ["circular top", "right-side connection", "descending stroke"],
                "topology": "one endpoint with one self-intersection"
            }
        }
        # Pre-defined angle concepts with enhanced descriptions
        self.angle_concepts = {
            0: {
                "base_desc": "standard upright orientation",
                "angle_desc": "digit appears in its natural, upright position",
                "visual_changes": "no rotation applied, maintaining original appearance",
                "spatial_relations": "top remains at top, bottom at bottom, preserving vertical alignment",
                "reference_points": "all features maintain their expected positions relative to viewing direction"
            },
            90: {
                "base_desc": "rotated right with horizontal alignment",
                "angle_desc": "digit is turned 90 degrees clockwise, lying on its side",
                "visual_changes": "vertical elements become horizontal, horizontal elements become vertical",
                "spatial_relations": "original top points right, original bottom points left",
                "reference_points": "leftmost point becomes highest point, rightmost becomes lowest"
            },
            180: {
                "base_desc": "inverted upside down",
                "angle_desc": "digit is completely flipped, appearing upside down",
                "visual_changes": "complete inversion of all elements, mirror image across horizontal axis",
                "spatial_relations": "top becomes bottom, bottom becomes top, left and right swap",
                "reference_points": "highest points become lowest, and vice versa"
            },
            270: {
                "base_desc": "rotated left with horizontal alignment",
                "angle_desc": "digit is turned 90 degrees counterclockwise, lying on its side",
                "visual_changes": "vertical elements become horizontal, horizontal elements become vertical",
                "spatial_relations": "original top points left, original bottom points right",
                "reference_points": "rightmost point becomes highest point, leftmost becomes lowest"
            }
        }
        
    def _get_embedding(self, text):
        """Get BERT embedding for a text prompt"""
        # Tokenize
        tokens = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )
        
        # Move to correct device
        tokens = {k: v.to(self.config.training.device) for k, v in tokens.items()}
        
        # Get embedding
        with torch.no_grad():
            outputs = self.llm(**tokens)
            last_hidden_state = outputs.last_hidden_state
            attention_mask = tokens['attention_mask']
            
            # Mean pooling
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
            embedding = torch.sum(last_hidden_state * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            
        return embedding
    
    def __call__(self, digit, angle):
        """Extract concepts for a digit-angle pair"""
        # Get concept descriptions
        digit_desc = self.digit_concepts.get(digit,{
            "primary": "unknown digit",
            "structural": ["unknown structure"],
            "components": ["unknown components"],
            "topology": "unknown topology"
        })
        invariant_prompt = self.invariant_template.format(
            digit=digit,
            primary=digit_desc["primary"],
            structural=", ".join(digit_desc["structural"]),
            components=", ".join(digit_desc["components"]),
            topology=digit_desc["topology"]
        )
        
        angle_desc = self.angle_concepts.get(angle,{
            "base_desc": "unknown angle",
            "angle_desc": "unknown rotation effect",
            "visual_changes": "unknown visual changes",
            "spatial_relations": "unknown spatial relations",
            "reference_points": "unknown reference points"
        })
        
        variant_prompt = self.variant_template.format(
            digit=digit,
            angle=angle,
            base_desc=angle_desc["base_desc"],
            angle_desc=angle_desc["angle_desc"]
        )
        
        # Encode with BERT
        with torch.no_grad():
            # Encode invariant (digit) concepts
            inv_tokens = self.tokenizer(invariant_prompt, return_tensors="pt", padding=True)
            inv_output = self.model(**inv_tokens)
            inv_concepts = inv_output.last_hidden_state[:, 0]  # CLS token
            
            # Encode variant (angle) concepts
            var_tokens = self.tokenizer(variant_prompt, return_tensors="pt", padding=True)
            var_output = self.model(**var_tokens)
            var_concepts = var_output.last_hidden_state[:, 0]  # CLS token
        
        return {
            'invariant_concepts': inv_concepts.squeeze(0),
            'variant_concepts': var_concepts.squeeze(0)
        }


class AdvancedLLMConceptExtractor:
    """More sophisticated concept extractor using API-based LLM"""
    def __init__(self, api_key=None):
        self.api_key = api_key
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.model = AutoModel.from_pretrained("bert-base-uncased")
        
        self.prompt_template = """
        Analyze a handwritten digit {digit} rotated by {angle} degrees.
        
        Provide two types of concepts:
        1. Invariant concepts (properties that don't change with rotation):
        2. Variant concepts (properties specific to this rotation):
        
        Format: JSON with 'invariant' and 'variant' lists
        """
    
    def __call__(self, digit, angle):
        # Format prompt
        prompt = self.prompt_template.format(digit=digit, angle=angle)
        
        try:
            # Get LLM response (implement based on your LLM API)
            concepts = self._get_llm_response(prompt)
            
            # Encode concepts
            inv_concepts = self._encode_concepts(concepts['invariant'])
            var_concepts = self._encode_concepts(concepts['variant'])
            
            return {
                'invariant_concepts': inv_concepts,
                'variant_concepts': var_concepts,
                'raw_concepts': concepts  # Original LLM output
            }
        except Exception as e:
            print(f"Error getting LLM concepts: {e}")
            # Fallback to simple concepts
            return SimpleLLMConceptExtractor()(digit, angle)
    
    def _encode_concepts(self, concept_list):
        # Join concepts into single string
        concept_text = " ".join(concept_list)
        
        # Encode with BERT
        with torch.no_grad():
            tokens = self.tokenizer(concept_text, return_tensors="pt", padding=True)
            outputs = self.model(**tokens)
            concepts = outputs.last_hidden_state[:, 0]  # CLS token
        
        return concepts.squeeze(0)

# Usage example
def get_dataset(root_dir,train=None,angles=None,concept_type='simple'):
    # Choose concept extractor
    if concept_type == 'simple':
        concept_extractor = SimpleLLMConceptExtractor(config)
    elif concept_type == 'advanced':
        concept_extractor = AdvancedLLMConceptExtractor()
    else:
        concept_extractor = None
    
   
    # Create dataset
    dataset = RotatedMNIST(
        root_dir=config.data.root_dir,
        config=config,
        train=True,
        angles=config.data.angles,
        concept_type=concept_type
    )
    
    return dataset

