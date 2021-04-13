//=============================================================================================
// Computer Graphics Sample Program: Ray-tracing-let
//=============================================================================================
#include "framework.h"

enum MaterialType {ROUGH, REFLECTIVE};
struct Material {
	vec3 ka, kd, ks;
	float  shininess;
	vec3 F0;
	MaterialType type;
	bool portalMaterial;
	vec3 pentagonCenter; /// portalos visszaverodes szamitasnal van ra szukseg
	Material(MaterialType t) {
		type = t;
		pentagonCenter = vec3(0, 0, 0);
	}
};

struct RoughMaterial : Material {
	RoughMaterial(vec3 _kd, vec3 _ks, float _shininess) : Material(ROUGH) {
		ka = _kd * M_PI;
		kd = _kd;
		ks = _ks; 
		shininess = _shininess;
		portalMaterial = false;
	}
};


vec3 operator/(vec3 num, vec3 denum) {
	return vec3(num.x / denum.x, num.y / denum.y, num.z / denum.z);
}
struct ReflectiveMaterial : Material
{
	ReflectiveMaterial(vec3 n, vec3 kappa, bool isPolral) : Material(REFLECTIVE) {
		vec3 one(1, 1, 1);
		F0 = ((n - one) * (n - one) + kappa * kappa) / ((n + one) * (n + one) + kappa * kappa);
		portalMaterial = isPolral;
	}
};

struct Hit {
	float t;
	vec3 position, normal;
	Material* material;
	Hit() { t = -1; }
};

struct Ray {
	vec3 start, dir;
	Ray(vec3 _start, vec3 _dir) { start = _start; dir = normalize(_dir); }
};

class Intersectable {
protected:
	Material* material;
public:
	virtual Hit intersect(const Ray& ray) = 0;
};

struct Sphere : public Intersectable {
	vec3 center;
	float radius;

	Sphere(const vec3& _center, float _radius, Material* _material) {
		center = _center;
		radius = _radius;
		material = _material;
	}

	Hit intersect(const Ray& ray) {
		Hit hit;
		vec3 dist = ray.start - center;
		float a = dot(ray.dir, ray.dir);
		float b = dot(dist, ray.dir) * 2.0f;
		float c = dot(dist, dist) - radius * radius;
		float discr = b * b - 4.0f * a * c;
		if (discr < 0) return hit;
		float sqrt_discr = sqrtf(discr);
		float t1 = (-b + sqrt_discr) / 2.0f / a;	// t1 >= t2 for sure
		float t2 = (-b - sqrt_discr) / 2.0f / a;
		if (t1 <= 0) return hit;
		hit.t = (t2 > 0) ? t2 : t1;
		hit.position = ray.start + ray.dir * hit.t;
		hit.normal = (hit.position - center) * (1.0f / radius);
		hit.material = material;
		return hit;
	}
};

struct GoldObject : public Intersectable {
	float aParam, bParam, cParam;
	GoldObject(float aParam, float bParam, float cParam) {
		this->aParam = aParam;
		this->bParam = bParam;
		this->cParam = cParam;
		vec3 n(0.17, 0.35, 1.5); vec3 kappa(3.1, 2.7, 1.9);
		material = new ReflectiveMaterial(n, kappa, false);
	}

	Hit intersect(const Ray& ray) {
		Hit hit;

		float a = aParam * pow(ray.dir.x, 2) + bParam * pow(ray.dir.y, 2);
		float b = 2.0f * aParam * ray.start.x * ray.dir.x + 2.0f * bParam * ray.start.y * ray.dir.y - cParam * ray.dir.z;
		float c = aParam * pow(ray.start.x, 2) + bParam * pow(ray.start.y, 2) - cParam * ray.start.z;
		float discr = b * b - 4.0f * a * c;
		if (discr < 0) return hit;
		float sqrt_discr = sqrtf(discr);
		float t1 = (-b + sqrt_discr) / 2.0f / a;	// t1 >= t2 for sure
		float t2 = (-b - sqrt_discr) / 2.0f / a;
		if (t1 <= 0) return hit;

		if (insideSphere(ray.start + ray.dir * t2)) {
			hit.t = t2;
		}
		else if(insideSphere(ray.start + ray.dir * t1)){
			hit.t = t1;
		}
		else {
			return hit; // nincsenek benne a 0.3 sugaru korben
		}
		//hit.t = (t2 > 0) ? t2 : t1;
		hit.position = ray.start + ray.dir * hit.t;

		float tmp = exp(aParam * pow(hit.position.x, 2) + bParam * pow(hit.position.y, 2) - cParam * hit.position.z);
		vec3 gradient = vec3( tmp * 2 * hit.position.x * aParam, tmp * 2 * hit.position.y * bParam, -cParam * tmp);
		//vec3 normal2 = normalize(vec3(-2 * aParam * hit.position.x / cParam, -2 * bParam * hit.position.y / cParam, 1));

		hit.normal = -normalize(gradient);
		//printf("n: %f %f %f\n", hit.normal.x, hit.normal.y, hit.normal.z);
		//printf("n2: %f %f %f\n", normal2.x, normal2.y, normal2.z);


		hit.material = material;
		return hit;
	
	}
	bool insideSphere(vec3 point) {
		if (sqrt(pow(point.x, 2) + pow(point.y, 2) + pow(point.z, 2)) < 0.3f) return true;
		return false;
	}
};

struct Face {
	vec3 points[5];
	vec3 normal() {
		return -normalize(cross(points[1] - points[0], points[2]-points[0])); // a sik ket vektorananak veszem a skalaris szorzatat es normalizalom
	}
};

struct Dodecahedron : Intersectable{
	Face faces[12];
	Material* material2;
	Material* material3;
	Material* material4;
	Material* material5;
	Material* material6;
	Material* material7;
	Material* material8;
	Material* material9;
	Material* material10;
	Material* material11;
	Material* material12;
	Material* material13;
	Dodecahedron() {
		vec3 kd(0.73f, 0.82f, 0.65f), ks(2, 2, 2);
		material = new RoughMaterial(kd, ks, 1000);

		

		vec3 n(1, 1, 1); vec3 kappa(10, 10, 10);
		material2 = new ReflectiveMaterial(n, kappa, true);

		vec3 kd3(0.5f, 0.5f, 0.2f), ks3(2, 2, 2);
		material3 = new RoughMaterial(kd3, ks3, 50);

		vec3 kd4(0.5f, 0.2f, 0.5f), ks4(2, 2, 2);
		material4 = new RoughMaterial(kd4, ks4, 50);

		vec3 kd5(0.5f, 0.2f, 0.9f), ks5(2, 2, 2);
		material5 = new RoughMaterial(kd5, ks5, 50);

		vec3 kd6(0.1f, 0.4f, 0.2f), ks6(2, 2, 2);
		material6 = new RoughMaterial(kd6, ks6, 50);

		vec3 kd7(0.1f, 0.2f, 0.6f), ks7(2, 2, 2);
		material7 = new RoughMaterial(kd7, ks7, 50);

		vec3 kd8(0.5f, 0.5f, 0.5f), ks8(2, 2, 2);
		material8 = new RoughMaterial(kd8, ks8, 50);

		vec3 kd9(0.2f, 0.6f, 0.2f), ks9(2, 2, 2);
		material9 = new RoughMaterial(kd9, ks9, 50);

		vec3 kd10(0.5f, 0.8f, 0.2f), ks10(2, 2, 2);
		material10 = new RoughMaterial(kd10, ks10, 50);

		vec3 kd11(0.8f, 0.1f, 0.3f), ks11(2, 2, 2);
		material11 = new RoughMaterial(kd11, ks11, 50);

		vec3 kd12(0.9f, 0.6f, 0.2f), ks12(2, 2, 2);
		material12 = new RoughMaterial(kd12, ks12, 50);

		vec3 kd13(0.9f, 0.7f, 0.5f), ks13(2, 2, 2);
		material13 = new RoughMaterial(kd13, ks13, 50);

		std::vector<vec3> objVertices;
		objVertices.push_back(vec3(0, 0.618, 1.618));
		objVertices.push_back(vec3(0, -0.618, 1.618));
		objVertices.push_back(vec3(0, -0.618, -1.618));
		objVertices.push_back(vec3(0, 0.618, -1.618));

		objVertices.push_back(vec3(1.618, 0, 0.618));
		objVertices.push_back(vec3(-1.618, 0, 0.618));
		objVertices.push_back(vec3(-1.618, 0, -0.618));
		objVertices.push_back(vec3(1.618, 0, -0.618));

		objVertices.push_back(vec3(0.618, 1.618, 0));
		objVertices.push_back(vec3(-0.618, 1.618, 0));
		objVertices.push_back(vec3(-0.618, -1.618, 0));
		objVertices.push_back(vec3(0.618, -1.618, 0));

		objVertices.push_back(vec3(1, 1, 1));
		objVertices.push_back(vec3(-1, 1, 1));
		objVertices.push_back(vec3(-1, -1, 1));
		objVertices.push_back(vec3(1, -1, 1));

		objVertices.push_back(vec3(1, -1, -1));
		objVertices.push_back(vec3(1, 1, -1));
		objVertices.push_back(vec3(-1, 1, -1));
		objVertices.push_back(vec3(-1, -1, -1));

		int indexes[12][5] = {	{1, 2, 16, 5, 13}, // egyes lapokhoz tartozo csucsok indexei (+1)
								{1, 13, 9, 10, 14},
								{1, 14, 6, 15, 2},
								{2, 15, 11, 12, 16},
								{3, 4, 18, 8, 17},
								{3, 17, 12, 11, 20},
								{3, 20, 7, 19, 4},
								{19, 10, 9, 18, 4},
								{16, 12, 17, 8, 5},
								{5, 8, 18, 9, 13},
								{14, 10, 19, 7, 6},
								{6, 7, 20, 11, 15} };

		for (int i = 0; i < 12; i++)
		{
			Face tmpFace;
			for (int j = 0; j < 5; j++)
			{
				tmpFace.points[j] = objVertices.at(indexes[i][j] - 1);
			}
			faces[i] = tmpFace;
		}
		
		//printFaces();
	}



	Hit intersect(const Ray& ray) {
		Hit hit;

		int bestFaceIndex = 0;
		float smallestPositiveT = -1;
		for (int i = 0; i < 12; i++)
		{
			float nx = faces[i].normal().x;
			float ny = faces[i].normal().y;
			float nz = faces[i].normal().z;
			float x0 = faces[i].points[0].x;
			float y0 = faces[i].points[0].y;
			float z0 = faces[i].points[0].z;
			float t = (nx * x0 + ny * y0 + nz * z0 - nx * ray.start.x - ny * ray.start.y - nz * ray.start.z) / (nx * ray.dir.x + ny * ray.dir.y + nz * ray.dir.z); 
			if (t > 0 && (smallestPositiveT < 0 || smallestPositiveT > t)) {
				smallestPositiveT = t;
				bestFaceIndex = i;
			}
		}
		if (smallestPositiveT > 0) {
			hit.t = smallestPositiveT;
			hit.position = ray.start + ray.dir * hit.t;
			hit.normal = faces[bestFaceIndex].normal();
			hit.material = material;
			if (closestSideDistance(bestFaceIndex, hit.position) > 0.05) {
				vec3 pentagonCenter = vec3(0, 0, 0); // sulypont lesz a kozeppont
				for (int i = 0; i < 5; i++)
				{
					pentagonCenter = pentagonCenter + faces[bestFaceIndex].points[i];
				}
				pentagonCenter = pentagonCenter / 5.0f;
		
				hit.material = material2;
				hit.material->pentagonCenter = pentagonCenter;

			}
			
			//if (bestFaceIndex == 2) hit.material = material2;
			//printf("dist %f\n", distanceFromLine(hit.position, faces[bestFaceIndex].points[0], faces[bestFaceIndex].points[1]));
			/*if (bestFaceIndex == 2) hit.material = material3;
			if (bestFaceIndex == 3) hit.material = material4;
			if (bestFaceIndex == 4) hit.material = material5;
			if (bestFaceIndex == 5) hit.material = material6;
			if (bestFaceIndex == 6) hit.material = material7;
			if (bestFaceIndex == 7) hit.material = material8;
			if (bestFaceIndex == 8) hit.material = material9;
			if (bestFaceIndex == 9) hit.material = material10;
			if (bestFaceIndex == 10) hit.material = material11;
			if (bestFaceIndex == 11) hit.material = material12;
			if (bestFaceIndex == 12) hit.material = material13;
			*/
		}
		//printf("hit %f face %f\n", hit.t, (float)bestFaceIndex);

		return hit;
	}
	void printFaces() {
		for (int i = 0; i < 12; i++)
		{
			vec3 normal = faces[i].normal();
			printf("\ni: %f normal %f %f %f\n", (float)i, normal.x, normal.y, normal.z);
			for (int j = 0; j < 5; j++)
			{
				printf("point: %f x: %f y: %f z: %f\n", (float)j + 1, faces[i].points[j].x, faces[i].points[j].y, faces[i].points[j].z);

			}
		}
	}

	float distanceFromLine(vec3 point, vec3 linePoint1, vec3 linePoint2) {
		return length(cross((linePoint2 - linePoint1), (linePoint1 - point))) / length(linePoint2 - linePoint1);
	}
	float closestSideDistance(int faceIndex, vec3 point) {
		float dist = -1;
		for (int i = 0; i < 4; i++) // elso negy egyenes
		{
			float tmpDist = distanceFromLine(point, faces[faceIndex].points[i], faces[faceIndex].points[i + 1]);
			if (tmpDist < dist || dist < 0) dist = tmpDist;
		}
		// otodik egyenes
		float tmpDist = distanceFromLine(point, faces[faceIndex].points[0], faces[faceIndex].points[4]);
		if (tmpDist < dist || dist < 0) dist = tmpDist;
		return dist;
	}

};

class Camera {
	vec3 eye, lookat, right, up;
	float fov;
public:
	void set(vec3 _eye, vec3 _lookat, vec3 vup, float fov) {
		eye = _eye;
		lookat = _lookat;
		vec3 w = eye - lookat;
		float focus = length(w);
		this->fov = fov;
		right = normalize(cross(vup, w)) * focus * tanf(fov / 2);
		up = normalize(cross(w, right)) * focus * tanf(fov / 2);
	}
	Ray getRay(int X, int Y) {
		vec3 dir = lookat + right * (2.0f * (X + 0.5f) / windowWidth - 1) + up * (2.0f * (Y + 0.5f) / windowHeight - 1) - eye;
		return Ray(eye, dir);
	}
	void Animate(float dt) {
		vec3 d = eye - lookat;
		//printf("eye x %f y %f z %f\n", eye.x, eye.y, eye.z);
		eye = vec3(d.x * cos(dt) + d.z * sin(dt), d.y, -d.x * sin(dt) + d.z * cos(dt)) + lookat;
		//printf(" eye x %f y %f z %f\n", eye.x, eye.y, eye.z);
		set(eye, lookat, up, fov);
	}
};

struct Light {
	vec3 direction;
	vec3 Le;
	Light(vec3 _direction, vec3 _Le) {
		direction = normalize(_direction);
		Le = _Le;
	}
};

struct PointLight {
	vec3 location;
	vec3 Le;
	PointLight(vec3 _location, vec3 _Le) {
		this->location = _location;
		Le = _Le;
	}
};

float rnd() { return (float)rand() / RAND_MAX; }

const float epsilon = 0.01f;

class Scene {
	std::vector<Intersectable*> objects;
	std::vector<Light*> lights;
	std::vector<PointLight*> pointLights;
	Camera camera;
	vec3 La;
public:
	void animate(float dt) {
		camera.Animate(dt);
	}
	void build() {
		vec3 eye = vec3(0, 0, 1.3), vup = vec3(0, 1, 0), lookat = vec3(0, 0, 0);
		float fov = 45 * M_PI / 180;
		camera.set(eye, lookat, vup, fov);

		La = vec3(0.1f, 0.1f, 0.1f);
		vec3 lightDirection(1, 1, 1), Le(2, 2, 2);
		//lights.push_back(new Light(lightDirection, Le));

		vec3 position(0, 0.6, 0), LePoint(1, 1, 1);
		pointLights.push_back(new PointLight(position, LePoint));

		vec3 kd(0.3f, 0.2f, 0.1f), ks(2, 2, 2);
		Material* material1 = new RoughMaterial(kd, ks, 50);

		vec3 n(0.17, 0.35, 1.5); vec3 kappa(3.1, 2.7, 1.9);
		Material* material2 = new ReflectiveMaterial(n, kappa, false);
	
		//objects.push_back(new Sphere(vec3(0.0f,0.0f, 0.0f), 0.2f, material2));
	


		objects.push_back(new Dodecahedron());

		objects.push_back(new GoldObject(5.0f, 5.0f, 1.0f));
		//objects.push_back(new Sphere(vec3(0.0f,0.0f, -2.0f), 0.2f, material1));
		//objects.push_back(new Sphere(vec3(2.0f, 0.0f, 0.0f), 0.2f, material1));
		//objects.push_back(new Sphere(vec3(-2.0f, 0.0f, 0.0f), 0.2f, material1));
		//objects.push_back(new Sphere(vec3(0.0f, 0.0f, 2.0f), 0.2f, material1));
		/*objects.push_back(new Sphere(vec3(0.6f,-0.2f, 0.0f), 0.2f, material1));
		objects.push_back(new Sphere(vec3(0.6f,0.0f, 1.5f), 0.2f, material1));
		objects.push_back(new Sphere(vec3(1.0f,1.0f, 2.0f), 0.2f, material1));
		objects.push_back(new Sphere(vec3(0.1f,0.5f, 2.0f), 0.4f, material1));
		*/
		
	}

	void render(std::vector<vec4>& image) {
		for (int Y = 0; Y < windowHeight; Y++) {
#pragma omp parallel for
			for (int X = 0; X < windowWidth; X++) {
				vec3 color = trace(camera.getRay(X, Y));
				image[Y * windowWidth + X] = vec4(color.x, color.y, color.z, 1);
			}
		}
	}

	Hit firstIntersect(Ray ray) {
		Hit bestHit;
		for (Intersectable* object : objects) {
			Hit hit = object->intersect(ray); //  hit.t < 0 if no intersection
			if (hit.t > 0 && (bestHit.t < 0 || hit.t < bestHit.t))  bestHit = hit;
		}
		if (dot(ray.dir, bestHit.normal) > 0) bestHit.normal = bestHit.normal * (-1);
		return bestHit;
	}

	bool shadowIntersect(Ray ray) {	// for directional lights
		for (Intersectable* object : objects) if (object->intersect(ray).t > 0) return true;
		return false;
	}
	bool shadowIntersectPointLight(Ray ray, float dist) {	// pontszeru fenyforrasra
		Hit shadowHit = firstIntersect(ray);
		if ((shadowHit.t < 0 || shadowHit.t > dist)) return true;
		return false;
	}

	vec3 rotatePoint(vec3 pointToRotate, vec3 rotationAxisNormal, vec3 pentagonCenter) {
		pointToRotate = pointToRotate - pentagonCenter; // eltolom az otszoget az origoba, ugy, hogy a kozeppontja legyen a (0,0,0) pontban
		float sinTheta = sin(0.4 * M_PI); // 72 fok radianban 0.4pi
		float cosTheta = cos(0.4 * M_PI);
		vec3 rotated = (pointToRotate * cosTheta) + (cross(rotationAxisNormal, pointToRotate) * sinTheta) + (rotationAxisNormal * dot(rotationAxisNormal, pointToRotate)) * (1 - cosTheta);
		rotated = rotated + pentagonCenter; // visszatoljuk a helyere
		return rotated;
	}
	vec3 rotateVector(vec3 pointToRotate, vec3 rotationAxisNormal) { // vektornal nincsen szukseg eltolasra, az az origobol mutat amugy is
		float sinTheta = sin(0.4 * M_PI); // 72 fok radianban 0.4pi
		float cosTheta = cos(0.4 * M_PI);
		vec3 rotated = (pointToRotate * cosTheta) + (cross(rotationAxisNormal, pointToRotate) * sinTheta) + (rotationAxisNormal * dot(rotationAxisNormal, pointToRotate)) * (1 - cosTheta);
		return rotated;
	}


	vec3 trace(Ray ray, int depth = 0, int portalDepth = 0) {
		if (depth > 5) return La;
		Hit hit = firstIntersect(ray);
		if (hit.t < 0) return La;
		vec3 outRadiance = hit.material->ka * La;
		if (hit.material->type == ROUGH) {
			for (Light* light : lights) { 
				Ray shadowRay(hit.position + hit.normal * epsilon, light->direction);
				float cosTheta = dot(hit.normal, light->direction);
				if (cosTheta > 0 && !shadowIntersect(shadowRay)) {	// shadow computation
					outRadiance = outRadiance + light->Le * hit.material->kd * cosTheta;
					vec3 halfway = normalize(-ray.dir + light->direction);
					float cosDelta = dot(hit.normal, halfway);
					if (cosDelta > 0) outRadiance = outRadiance + light->Le * hit.material->ks * powf(cosDelta, hit.material->shininess);
				}
			}
			
				for (PointLight* pointLight : pointLights) { // pontszeru lampakra
					vec3 lightVector = normalize( pointLight->location - hit.position); // a feluletrol a pontszeru lampaba mutato vektor normalizaltja
					float distanceFromPointLight = length(pointLight->location - hit.position); // tavolsag a lampatol (fenyerosseg negyzetesen csokken)
					Ray shadowRay(hit.position + hit.normal * epsilon, lightVector);
					
					float cosTheta = dot(hit.normal, lightVector);
					if (cosTheta > 0 && shadowIntersectPointLight(shadowRay, distanceFromPointLight)) {	// shadow computation
						vec3 Le = pointLight->Le * (1.0f / pow(distanceFromPointLight, 2)); // forditottan aranyos a tavolsag negyzetevel
						outRadiance = outRadiance + Le * hit.material->kd * cosTheta; // diffuz

						vec3 halfway = normalize(-ray.dir + lightVector); //csillogas
						float cosDelta = dot(hit.normal, halfway);
						if (cosDelta > 0) outRadiance = outRadiance + Le * hit.material->ks * powf(cosDelta, hit.material->shininess);
					}
				}


		}
		if (hit.material->type == REFLECTIVE) {
			if (hit.material->portalMaterial) {
				vec3 reflectedDir = ray.dir - hit.normal * dot(hit.normal, ray.dir) * 2.0f;
				float cosa = -dot(ray.dir, hit.normal);
				vec3 one(1, 1, 1);
				vec3 F = hit.material->F0 + (one - hit.material->F0) * pow(1 - cosa, 5);

				vec3 newStartPoint = rotatePoint(hit.position, hit.normal, hit.material->pentagonCenter);
				vec3 newDirection = rotateVector(reflectedDir, hit.normal);
				outRadiance = outRadiance + F * trace(Ray(newStartPoint + hit.normal * epsilon, newDirection), depth + 1);
			}
			else {
				vec3 reflectedDir = ray.dir - hit.normal * dot(hit.normal, ray.dir) * 2.0f;
				float cosa = -dot(ray.dir, hit.normal);
				vec3 one(1, 1, 1);
				vec3 F = hit.material->F0 + (one - hit.material->F0) * pow(1 - cosa, 5);
				outRadiance = outRadiance + F * trace(Ray(hit.position + hit.normal * epsilon, reflectedDir), depth + 1);
			}
				
		}

		return outRadiance;
	}
};

GPUProgram gpuProgram; // vertex and fragment shaders
Scene scene;

// vertex shader in GLSL
const char* vertexSource = R"(
	#version 330
    precision highp float;

	layout(location = 0) in vec2 cVertexPosition;	// Attrib Array 0
	out vec2 texcoord;

	void main() {
		texcoord = (cVertexPosition + vec2(1, 1))/2;							// -1,1 to 0,1
		gl_Position = vec4(cVertexPosition.x, cVertexPosition.y, 0, 1); 		// transform to clipping space
	}
)";

// fragment shader in GLSL
const char* fragmentSource = R"(
	#version 330
    precision highp float;

	uniform sampler2D textureUnit;
	in  vec2 texcoord;			// interpolated texture coordinates
	out vec4 fragmentColor;		// output that goes to the raster memory as told by glBindFragDataLocation

	void main() {
		fragmentColor = texture(textureUnit, texcoord); 
	}
)";

class FullScreenTexturedQuad {
	unsigned int vao;	// vertex array object id and texture id
	
	int windowWidth, windowHeight;
public:
	FullScreenTexturedQuad(int windowWidth, int windowHeight)
	{
		this->windowWidth = windowWidth;
		this->windowHeight = windowHeight;
		glGenVertexArrays(1, &vao);	// create 1 vertex array object
		glBindVertexArray(vao);		// make it active

		unsigned int vbo;		// vertex buffer objects
		glGenBuffers(1, &vbo);	// Generate 1 vertex buffer objects

		// vertex coordinates: vbo0 -> Attrib Array 0 -> vertexPosition of the vertex shader
		glBindBuffer(GL_ARRAY_BUFFER, vbo); // make it active, it is an array
		float vertexCoords[] = { -1, -1,  1, -1,  1, 1,  -1, 1 };	// two triangles forming a quad
		glBufferData(GL_ARRAY_BUFFER, sizeof(vertexCoords), vertexCoords, GL_STATIC_DRAW);	   // copy to that part of the memory which is not modified 
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, NULL);     // stride and offset: it is tightly packed
	}

	void Draw(std::vector<vec4>& image){
		Texture texture(windowWidth, windowHeight, image);
		glBindVertexArray(vao);	// make the vao and its vbos active playing the role of the data source
		gpuProgram.setUniform(texture, "textureUnit");
		glDrawArrays(GL_TRIANGLE_FAN, 0, 4);	// draw two triangles forming a quad
	}
};

FullScreenTexturedQuad* fullScreenTexturedQuad;

// Initialization, create an OpenGL context
void onInitialization() {
	glViewport(0, 0, windowWidth, windowHeight);
	scene.build();

	
	//long timeStart = glutGet(GLUT_ELAPSED_TIME);
	
	//long timeEnd = glutGet(GLUT_ELAPSED_TIME);
	//printf("Rendering time: %d milliseconds\n", (timeEnd - timeStart));

	// copy image to GPU as a texture
	fullScreenTexturedQuad = new FullScreenTexturedQuad(windowWidth, windowHeight);

	// create program for the GPU
	gpuProgram.create(vertexSource, fragmentSource, "fragmentColor");
}

// Window has become invalid: Redraw
void onDisplay() {
	std::vector<vec4> image(windowWidth * windowHeight);
	scene.render(image);
	fullScreenTexturedQuad->Draw(image);
	glutSwapBuffers();									// exchange the two buffers
}

// Key of ASCII code pressed
void onKeyboard(unsigned char key, int pX, int pY) {
}

// Key of ASCII code released
void onKeyboardUp(unsigned char key, int pX, int pY) {

}

// Mouse click event
void onMouse(int button, int state, int pX, int pY) {
	if(state == 0) scene.animate(0.1f);
	
	glutPostRedisplay();
}

// Move mouse with key pressed
void onMouseMotion(int pX, int pY) {
}

// Idle event indicating that some time elapsed: do animation here
void onIdle() {
	//scene.animate(0.1f);

	//glutPostRedisplay();
}